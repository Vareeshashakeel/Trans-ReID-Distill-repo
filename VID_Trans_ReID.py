import argparse
import os
import random
import time

import numpy as np
import torch
from torch.cuda import amp

try:
    from torch_ema import ExponentialMovingAverage
except Exception:
    ExponentialMovingAverage = None

from Dataloader import dataloader
from Loss_fun import make_loss
from VID_Test import test
from VID_Trans_model import VID_Trans
from loss.distill_loss import FeatureDistillLoss, RelationDistillLoss
from loss.xcamera_supcon import CrossCameraSupConLoss
from utility import AverageMeter, optimizer, scheduler


def set_seed(seed=1234):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def parse_block_indices(text):
    if isinstance(text, (list, tuple)):
        return tuple(int(x) for x in text)
    return tuple(sorted({int(x.strip()) for x in str(text).split(',') if x.strip()}))


def load_teacher_weights(model, path):
    state = torch.load(path, map_location='cpu')
    if 'model' in state:
        state = state['model']
    if 'state_dict' in state:
        state = state['state_dict']
    clean = {}
    for k, v in state.items():
        clean[k.replace('module.', '')] = v
    missing, unexpected = model.load_state_dict(clean, strict=False)
    print('[INFO] Teacher weights loaded from:', path)
    print('[INFO] Teacher missing keys:', len(missing))
    print('[INFO] Teacher unexpected keys:', len(unexpected))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Camera-aware teacher -> camera-agnostic student distillation over best intermediate XCam model')
    parser.add_argument('--Dataset_name', required=True, type=str)
    parser.add_argument('--model_path', required=True, type=str, help='ViT/ImageNet pretrained weight path for student init')
    parser.add_argument('--teacher_path', required=True, type=str, help='trained camera-aware teacher checkpoint')
    parser.add_argument('--output_dir', default='./output_xcam_teacher_distill', type=str)
    parser.add_argument('--epochs', default=120, type=int)
    parser.add_argument('--eval_every', default=10, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seq_len', default=4, type=int)
    parser.add_argument('--center_w', default=0.0, type=float)
    parser.add_argument('--attn_w', default=1.0, type=float)
    parser.add_argument('--xcam_w', default=0.12, type=float)
    parser.add_argument('--xcam_temp', default=0.07, type=float)
    parser.add_argument('--xcam_same_cam_w', default=0.25, type=float)
    parser.add_argument('--xcam_blocks', default='8', type=str)
    parser.add_argument('--fd_w', default=0.20, type=float, help='feature distillation weight')
    parser.add_argument('--rd_w', default=0.10, type=float, help='relation distillation weight')
    parser.add_argument('--teacher_use_bn', action='store_true', help='distill teacher BN feature instead of raw global feature')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(1234)
    xcam_blocks = parse_block_indices(args.xcam_blocks)

    train_loader, _, num_classes, camera_num, view_num, q_val_loader, g_val_loader = dataloader(
        args.Dataset_name, batch_size=args.batch_size, num_workers=args.num_workers, seq_len=args.seq_len
    )

    student = VID_Trans(num_classes=num_classes, camera_num=camera_num, pretrainpath=args.model_path,
                        xcam_block_indices=xcam_blocks, camera_aware=False)
    teacher = VID_Trans(num_classes=num_classes, camera_num=camera_num, pretrainpath=args.model_path,
                        xcam_block_indices=xcam_blocks, camera_aware=True)
    load_teacher_weights(teacher, args.teacher_path)

    loss_fun, center_criterion = make_loss(num_classes=num_classes)
    xcam_criterion = CrossCameraSupConLoss(temperature=args.xcam_temp, same_cam_weight=args.xcam_same_cam_w)
    feat_distill = FeatureDistillLoss()
    rel_distill = RelationDistillLoss()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    student = student.to(device)
    teacher = teacher.to(device)
    center_criterion = center_criterion.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    optimizer_main = optimizer(student)
    lr_scheduler = scheduler(optimizer_main)
    scaler = amp.GradScaler(enabled=(device == 'cuda'))
    optimizer_center = None
    if args.center_w > 0:
        optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=0.5)

    loss_meter = AverageMeter(); id_meter = AverageMeter(); xcam_meter = AverageMeter(); acc_meter = AverageMeter()
    fd_meter = AverageMeter(); rd_meter = AverageMeter()
    ema = ExponentialMovingAverage(student.parameters(), decay=0.995) if ExponentialMovingAverage is not None else None
    best_rank1 = 0.0

    print(f'[INFO] Device: {device}')
    print(f'[INFO] Student XCam blocks: {xcam_blocks}')
    print(f'[INFO] Loss weights -> center: {args.center_w}, attn: {args.attn_w}, xcam: {args.xcam_w}, fd: {args.fd_w}, rd: {args.rd_w}')
    print('[INFO] Distillation: camera-aware teacher -> camera-agnostic student')

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        for m in [loss_meter, id_meter, xcam_meter, acc_meter, fd_meter, rd_meter]:
            m.reset()
        lr_scheduler.step(epoch)
        student.train()
        teacher.eval()

        for iteration, (img, pid, target_cam, labels2) in enumerate(train_loader, start=1):
            optimizer_main.zero_grad()
            if optimizer_center is not None:
                optimizer_center.zero_grad()

            img = img.to(device)
            pid = pid.to(device)
            target_cam = target_cam.to(device)
            seq_cam = target_cam[:, 0].contiguous() if target_cam.ndim > 1 else target_cam.contiguous()
            labels2 = labels2.to(device)

            with torch.no_grad():
                _, _, _, teacher_aux = teacher(img, pid, cam_label=seq_cam)
                teacher_feat = teacher_aux['global_bn'] if args.teacher_use_bn else teacher_aux['global_raw']

            with amp.autocast(enabled=(device == 'cuda')):
                score, feat, a_vals, aux = student(img, pid, cam_label=seq_cam)
                attn_noise = a_vals * labels2
                attn_loss = attn_noise.sum(1).mean()
                loss_id, center = loss_fun(score, feat, pid, seq_cam)
                xcam_loss = xcam_criterion(aux['xcam_feats'], pid, seq_cam)
                student_feat = aux['global_raw']
                fd_loss = feat_distill(student_feat, teacher_feat)
                rd_loss = rel_distill(student_feat, teacher_feat)
                loss = loss_id + args.center_w * center + args.attn_w * attn_loss + args.xcam_w * xcam_loss + args.fd_w * fd_loss + args.rd_w * rd_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer_main)
            if optimizer_center is not None:
                for param in center_criterion.parameters():
                    if param.grad is not None:
                        param.grad.data *= (1.0 / args.center_w)
                scaler.step(optimizer_center)
            scaler.update()
            if ema is not None:
                ema.update()

            if isinstance(score, list):
                acc = (score[0].max(1)[1] == pid).float().mean()
            else:
                acc = (score.max(1)[1] == pid).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            id_meter.update(loss_id.item(), img.shape[0])
            xcam_meter.update(xcam_loss.item(), img.shape[0])
            fd_meter.update(fd_loss.item(), img.shape[0])
            rd_meter.update(rd_loss.item(), img.shape[0])
            acc_meter.update(acc.item(), 1)

            if device == 'cuda':
                torch.cuda.synchronize()
            if iteration % 50 == 0:
                print('Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, ID+Tri: {:.3f}, XCam: {:.3f}, FD: {:.3f}, RD: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}'.format(
                    epoch, iteration, len(train_loader), loss_meter.avg, id_meter.avg, xcam_meter.avg,
                    fd_meter.avg, rd_meter.avg, acc_meter.avg, lr_scheduler._get_lr(epoch)[0]))

        print('Epoch {} finished in {:.1f}s'.format(epoch, time.time() - start_time))
        if epoch % args.eval_every == 0:
            student.eval()
            rank1, mAP = test(student, q_val_loader, g_val_loader)
            print('CMC: %.4f, mAP : %.4f' % (rank1, mAP))
            latest_path = os.path.join(args.output_dir, f'{args.Dataset_name}_xcam_teacher_distill_latest.pth')
            torch.save(student.state_dict(), latest_path)
            if best_rank1 < rank1:
                best_rank1 = rank1
                best_path = os.path.join(args.output_dir, f'{args.Dataset_name}_xcam_teacher_distill_best.pth')
                torch.save(student.state_dict(), best_path)
                print(f'[OK] Saved best checkpoint: {best_path}')
