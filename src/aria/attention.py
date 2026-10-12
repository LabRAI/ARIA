"""Optional FlexAttention tiling for devices with limited shared memory."""
from __future__ import annotations


def configure_flex_attention(block_size: int) -> None:
    """Keep Transformers' softcap/masking semantics and bound Triton tile storage.

    Gemma-2's 256-dimensional heads exceed the Blackwell workstation shared
    memory limit under some PyTorch 2.8 default tiles. The explicit tiles apply
    to both forward and backward kernels, including frozen reference models.
    """
    if block_size not in {16, 32, 64}:
        raise ValueError("flex block size must be 16, 32, or 64")
    from transformers import AttentionInterface
    from transformers.integrations.flex_attention import flex_attention_forward

    defaults = {key: block_size for key in ("BLOCK_M", "BLOCK_N", "BLOCK_M1", "BLOCK_N1", "BLOCK_M2", "BLOCK_N2")}
    defaults["num_stages"] = 1

    def bounded_flex_attention(*args, **kwargs):
        kwargs["kernel_options"] = {**defaults, **(kwargs.get("kernel_options") or {})}
        return flex_attention_forward(*args, **kwargs)

    AttentionInterface.register("flex_attention", bounded_flex_attention)
