"""FP32 baseline and native CUDA BF16 mixed precision (FP32 master weights)."""
import torch


def validate_precision(precision, device):
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16.")
    device = torch.device(device)
    if precision == "bf16":
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("BF16 requires a CUDA GPU with native BF16 support; use fp32 otherwise.")
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported(including_emulation=False):
                raise ValueError("Selected CUDA GPU has no native BF16 support; use fp32.")


def autocast_context(precision, device):
    """Validate even direct training/scoring calls; never silently fall back."""
    validate_precision(precision, device)
    if precision == "fp32":
        # Disable any enclosing autocast so explicit FP32 always means FP32.
        return torch.autocast(device_type=torch.device(device).type, enabled=False)
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)
