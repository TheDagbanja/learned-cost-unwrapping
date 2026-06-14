"""Model components: Swin-T+LoRA backbone, U-Net decoder, and the unwrapper.

``timm`` is imported lazily (only when the backbone is instantiated), so these
imports are safe on machines without the heavy deps installed.
"""

from .backbone import (
    LoRAAdapter,
    LoRAFusedQKV,
    SwinLoRABackbone,
    inject_lora_qkv,
    iter_lora_parameters,
    mark_only_lora_as_trainable,
)
from .decoder import UNetDecoder
from .unwrapper import PhaseUnwrapper, compute_prototypes, prototypical_logits

__all__ = [
    "LoRAAdapter",
    "LoRAFusedQKV",
    "SwinLoRABackbone",
    "inject_lora_qkv",
    "iter_lora_parameters",
    "mark_only_lora_as_trainable",
    "UNetDecoder",
    "PhaseUnwrapper",
    "compute_prototypes",
    "prototypical_logits",
]
