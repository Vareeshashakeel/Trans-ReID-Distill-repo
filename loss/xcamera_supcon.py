import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossCameraSupConLoss(nn.Module):
    """
    Supervised contrastive loss that prioritizes same-ID / different-camera positives.
    If an anchor has no cross-camera positives in the batch, it falls back to same-ID
    same-camera positives with a smaller weight so training stays numerically stable.
    """

    def __init__(self, temperature=0.07, same_cam_weight=0.25, eps=1e-12):
        super().__init__()
        self.temperature = temperature
        self.same_cam_weight = same_cam_weight
        self.eps = eps

    def forward(self, features, labels, camids):
        if isinstance(features, (list, tuple)):
            losses = [self._single_forward(feat, labels, camids) for feat in features]
            valid_losses = [loss for loss in losses if loss is not None]
            if not valid_losses:
                return labels.new_tensor(0.0, dtype=torch.float32)
            return sum(valid_losses) / len(valid_losses)
        loss = self._single_forward(features, labels, camids)
        if loss is None:
            return labels.new_tensor(0.0, dtype=torch.float32)
        return loss

    def _single_forward(self, features, labels, camids):
        if features.ndim != 2:
            raise ValueError(f"Expected [B, C] features, got shape {tuple(features.shape)}")
        batch_size = features.size(0)
        if batch_size < 2:
            return None

        features = F.normalize(features, dim=1)
        labels = labels.contiguous().view(-1, 1)
        camids = camids.contiguous().view(-1, 1)

        logits = torch.matmul(features, features.t()) / self.temperature
        logits = logits - logits.max(dim=1, keepdim=True)[0].detach()

        same_id = torch.eq(labels, labels.t())
        same_cam = torch.eq(camids, camids.t())
        diff_cam = ~same_cam
        eye = torch.eye(batch_size, device=features.device, dtype=torch.bool)

        cross_cam_pos = same_id & diff_cam & ~eye
        same_cam_pos = same_id & same_cam & ~eye
        valid_neg = ~same_id

        exp_logits = torch.exp(logits) * (~eye)
        denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(self.eps)
        log_prob = logits - torch.log(denom)

        cross_weight = cross_cam_pos.float()
        same_weight = same_cam_pos.float() * self.same_cam_weight
        pos_weight = cross_weight + same_weight
        pos_count = pos_weight.sum(dim=1)
        valid_anchor = pos_count > 0

        if valid_anchor.sum() == 0:
            return None

        mean_log_prob_pos = (pos_weight * log_prob).sum(dim=1) / pos_count.clamp_min(self.eps)
        loss = -mean_log_prob_pos[valid_anchor].mean()
        return loss
