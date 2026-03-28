# VID-Trans-ReID XCam + Camera-Aware Teacher Distillation

This repo keeps the **best single intermediate XCam student** and adds **camera-aware teacher -> camera-agnostic student distillation**.

## What changes
- student stays **camera-agnostic at inference**
- teacher is **camera-aware during training only**
- keep the successful **single intermediate XCam at block 8**
- add **light feature distillation** and **light relation distillation** from teacher to student
- avoid heavy end-stage add-ons that previously hurt Rank-1

## Recommended command
```bash
python VID_Trans_ReID.py \
  --Dataset_name Mars \
  --model_path "$VIT" \
  --teacher_path /path/to/camera_aware_best.pth \
  --output_dir ./output_xcam_teacher_distill \
  --epochs 120 \
  --eval_every 10 \
  --batch_size 64 \
  --num_workers 4 \
  --seq_len 4 \
  --center_w 0.0 \
  --attn_w 1.0 \
  --xcam_w 0.12 \
  --xcam_temp 0.07 \
  --xcam_same_cam_w 0.25 \
  --xcam_blocks 8 \
  --fd_w 0.20 \
  --rd_w 0.10
```

## Notes
- `teacher_path` should point to your **trained camera-aware checkpoint**.
- the saved best checkpoint is the **student only**.
- at test/inference, only the **camera-agnostic student** is used.
