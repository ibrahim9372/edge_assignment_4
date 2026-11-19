"""
Single entry point to build any model in the project from a name (+ config) and to
restore it from one of our self-describing checkpoints. Used by every inference,
export and benchmarking script, which is what makes the inference pipeline unified.
"""
import torch


def build_model(arch: str, num_classes: int, config: dict = None):
    from q1_kd.models import ResNet50, STUDENTS
    if arch == "resnet50":
        return ResNet50(num_classes)
    if arch in STUDENTS:
        return STUDENTS[arch](num_classes)
    if arch == "nas":
        from q2_nas.generator import build_from_config
        return build_from_config(config, num_classes)
    raise ValueError(f"unknown architecture '{arch}'")


def load_checkpoint(path: str, device="cpu"):
    """Returns (model in eval mode, checkpoint dict without the state_dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["arch"], ckpt["num_classes"], ckpt.get("config"))
    # load_state_dict casts fp16-stored weights (teacher) back to fp32 automatically
    model.load_state_dict(ckpt.pop("state_dict"))
    return model.to(device).eval(), ckpt
