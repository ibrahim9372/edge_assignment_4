"""
Knowledge-distillation criterion.

    L = alpha * CE(z_s, y)                                   (hard labels)
      + (1 - alpha) * T^2 * KL( softmax(z_t / T) || softmax(z_s / T) )   (soft targets)
      + beta * || r(F_s) - F_t ||^2                           (feature / hint loss)

* Soft targets (Hinton et al., 2015): the frozen teacher's logits are softened with
  temperature T so the student also learns the *relative* similarities between
  wrong classes ("dark knowledge"). The T^2 factor keeps gradient magnitudes of the
  soft term comparable when T changes.
* Feature distillation (FitNets, Romero et al., 2015): an intermediate student
  feature map F_s is projected by a small trainable regressor r (1x1 conv + BN) to
  the teacher's channel count and matched to the teacher's layer3 map F_t with MSE.
  If spatial sizes differ, both maps are average-pooled to the smaller size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_target_loss(student_logits, teacher_logits, T):
    """Temperature-scaled KL divergence between teacher and student distributions."""
    log_p_s = F.log_softmax(student_logits / T, dim=1)
    p_t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (T * T)


class FeatureAdapter(nn.Module):
    """1x1 conv regressor mapping student channels -> teacher channels."""

    def __init__(self, c_student, c_teacher):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(c_student, c_teacher, 1, bias=False), nn.BatchNorm2d(c_teacher))

    def forward(self, x):
        return self.proj(x)


def feature_loss(f_s, f_t):
    if f_s.shape[-1] != f_t.shape[-1]:
        size = min(f_s.shape[-1], f_t.shape[-1])
        f_s, f_t = F.adaptive_avg_pool2d(f_s, size), F.adaptive_avg_pool2d(f_t, size)
    return F.mse_loss(f_s.float(), f_t.float())


class KDCriterion(nn.Module):
    """Criterion(model, x, y) -> (loss, logits, parts) used by common.train_loop.fit."""

    def __init__(self, teacher, c_student, c_teacher, T=4.0, alpha=0.3, beta=1.0,
                 use_soft=True, use_feature=True):
        super().__init__()
        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.T, self.alpha, self.beta = T, alpha, beta
        self.use_soft, self.use_feature = use_soft, use_feature
        self.adapter = FeatureAdapter(c_student, c_teacher) if use_feature else None

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()  # teacher BN statistics must stay frozen
        return self

    def forward(self, model, x, y):
        with torch.no_grad():
            t_logits, t_feat = self.teacher(x, return_features=True)
        s_logits, s_feat = model(x, return_features=True)

        ce = F.cross_entropy(s_logits.float(), y)
        parts = {"ce": ce.item()}
        if self.use_soft:
            kd = soft_target_loss(s_logits.float(), t_logits.float(), self.T)
            loss = self.alpha * ce + (1 - self.alpha) * kd
            parts["kd"] = kd.item()
        else:
            loss = ce
        if self.use_feature:
            fl = feature_loss(self.adapter(s_feat), t_feat)
            loss = loss + self.beta * fl
            parts["feat"] = fl.item()
        return loss, s_logits, parts
