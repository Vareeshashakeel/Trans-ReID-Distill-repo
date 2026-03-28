import copy
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F

from vit_ID import Block, TransReID


def TCSS(features, shift, b, t):
    features = features.view(b, features.size(1), t * features.size(2))
    token = features[:, 0:1]
    batchsize = features.size(0)
    dim = features.size(-1)
    features = torch.cat([features[:, shift:], features[:, 1:shift]], dim=1)
    try:
        features = features.view(batchsize, 2, -1, dim)
    except Exception:
        features = torch.cat([features, features[:, -2:-1, :]], dim=1)
        features = features.view(batchsize, 2, -1, dim)
    features = torch.transpose(features, 1, 2).contiguous()
    features = features.view(batchsize, -1, dim)
    return features, token


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1 and m.affine:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class VID_Trans(nn.Module):
    def __init__(self, num_classes, camera_num=None, pretrainpath=None, xcam_block_indices=(8,),
                 xcam_use_patch_mean=True, camera_aware=False):
        super().__init__()
        self.in_planes = 768
        self.num_classes = num_classes
        self.xcam_block_indices = tuple(sorted(set(xcam_block_indices)))
        self.xcam_use_patch_mean = xcam_use_patch_mean
        self.camera_aware = camera_aware

        self.base = TransReID(
            img_size=[256, 128], patch_size=16, stride_size=[16, 16],
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
            drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            camera_num=(camera_num or 0), use_camera_embed=camera_aware,
        )
        if pretrainpath is not None:
            state_dict = torch.load(pretrainpath, map_location='cpu')
            self.base.load_param(state_dict, load=True)

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(copy.deepcopy(block), copy.deepcopy(layer_norm))

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        dpr = [x.item() for x in torch.linspace(0, 0, 12)]
        self.block1 = Block(dim=3072, num_heads=12, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                            drop=0, attn_drop=0, drop_path=dpr[11],
                            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.b2 = nn.Sequential(self.block1, nn.LayerNorm(3072))

        self.bottleneck_1 = nn.BatchNorm1d(3072); self.bottleneck_1.bias.requires_grad_(False); self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(3072); self.bottleneck_2.bias.requires_grad_(False); self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(3072); self.bottleneck_3.bias.requires_grad_(False); self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(3072); self.bottleneck_4.bias.requires_grad_(False); self.bottleneck_4.apply(weights_init_kaiming)

        self.classifier_1 = nn.Linear(3072, self.num_classes, bias=False); self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(3072, self.num_classes, bias=False); self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(3072, self.num_classes, bias=False); self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(3072, self.num_classes, bias=False); self.classifier_4.apply(weights_init_classifier)

        self.middle_dim = 256
        self.attention_conv = nn.Conv2d(self.in_planes, self.middle_dim, [1, 1])
        self.attention_tconv = nn.Conv1d(self.middle_dim, 1, 3, padding=1)
        self.attention_conv.apply(weights_init_kaiming)
        self.attention_tconv.apply(weights_init_kaiming)
        self.shift_num = 5

    def _aggregate_temporal_feature(self, token_tensor, b, t):
        token_tensor = self.base.norm(token_tensor)
        cls_feat = token_tensor[:, 0].view(b, t, -1)
        if self.xcam_use_patch_mean:
            patch_feat = token_tensor[:, 1:].mean(dim=1).view(b, t, -1)
            seq_feat = 0.5 * (cls_feat + patch_feat)
        else:
            seq_feat = cls_feat
        return seq_feat.mean(dim=1)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        b = x.size(0)
        t = x.size(1)
        x = x.view(x.size(0) * x.size(1), x.size(2), x.size(3), x.size(4))
        flat_cam = None
        if cam_label is not None and self.camera_aware:
            if cam_label.ndim > 1:
                flat_cam = cam_label[:, 0].contiguous().repeat_interleave(t)
            else:
                flat_cam = cam_label.contiguous().repeat_interleave(t)

        features, intermediate_tokens = self.base(x, cam_label=flat_cam, return_intermediate=True, out_indices=self.xcam_block_indices)
        b1_feat = self.b1(features)
        global_feat = b1_feat[:, 0]

        global_feat_4d = global_feat.unsqueeze(2).unsqueeze(3)
        a = F.relu(self.attention_conv(global_feat_4d))
        a = a.view(b, t, self.middle_dim).permute(0, 2, 1)
        a = F.relu(self.attention_tconv(a)).view(b, t)
        a_vals = a
        a = F.softmax(a, dim=1)
        x_tmp = global_feat.view(b, t, -1)
        a = a.unsqueeze(-1).expand(b, t, self.in_planes)
        att_x = torch.mul(x_tmp, a).sum(1)
        global_feat = att_x.view(b, self.in_planes)
        feat = self.bottleneck(global_feat)

        feature_length = features.size(1) - 1
        patch_length = feature_length // 4
        x_parts, token = TCSS(features, self.shift_num, b, t)

        part1_f = self.b2(torch.cat((token, x_parts[:, :patch_length]), dim=1))[:, 0]
        part2_f = self.b2(torch.cat((token, x_parts[:, patch_length:patch_length * 2]), dim=1))[:, 0]
        part3_f = self.b2(torch.cat((token, x_parts[:, patch_length * 2:patch_length * 3]), dim=1))[:, 0]
        part4_f = self.b2(torch.cat((token, x_parts[:, patch_length * 3:patch_length * 4]), dim=1))[:, 0]

        part1_bn = self.bottleneck_1(part1_f)
        part2_bn = self.bottleneck_2(part2_f)
        part3_bn = self.bottleneck_3(part3_f)
        part4_bn = self.bottleneck_4(part4_f)

        xcam_feats = [self._aggregate_temporal_feature(token_tensor, b, t) for _, token_tensor in intermediate_tokens]

        if self.training:
            Global_ID = self.classifier(feat)
            Local_ID1 = self.classifier_1(part1_bn)
            Local_ID2 = self.classifier_2(part2_bn)
            Local_ID3 = self.classifier_3(part3_bn)
            Local_ID4 = self.classifier_4(part4_bn)
            aux = {
                'xcam_feats': xcam_feats,
                'xcam_blocks': list(self.xcam_block_indices),
                'global_raw': global_feat,
                'global_bn': feat,
                'local_raw': [part1_f, part2_f, part3_f, part4_f],
                'retrieval_feat': torch.cat([feat, part1_bn / 4, part2_bn / 4, part3_bn / 4, part4_bn / 4], dim=1),
            }
            return [Global_ID, Local_ID1, Local_ID2, Local_ID3, Local_ID4], [global_feat, part1_f, part2_f, part3_f, part4_f], a_vals, aux
        return torch.cat([feat, part1_bn / 4, part2_bn / 4, part3_bn / 4, part4_bn / 4], dim=1)

