import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureDistillLoss(nn.Module):
    """Cosine feature distillation on global retrieval features."""

    def __init__(self, detach_teacher=True):
        super().__init__()
        self.detach_teacher = detach_teacher

    def forward(self, student_feat, teacher_feat):
        if self.detach_teacher:
            teacher_feat = teacher_feat.detach()
        s = F.normalize(student_feat, dim=1)
        t = F.normalize(teacher_feat, dim=1)
        return (1.0 - (s * t).sum(dim=1)).mean()


class RelationDistillLoss(nn.Module):
    """Batch relation distillation on pairwise cosine similarity matrices."""

    def __init__(self, detach_teacher=True):
        super().__init__()
        self.detach_teacher = detach_teacher

    def forward(self, student_feat, teacher_feat):
        if self.detach_teacher:
            teacher_feat = teacher_feat.detach()
        s = F.normalize(student_feat, dim=1)
        t = F.normalize(teacher_feat, dim=1)
        s_rel = s @ s.t()
        t_rel = t @ t.t()
        return F.mse_loss(s_rel, t_rel)
