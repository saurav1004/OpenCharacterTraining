"""Python-only stub for flash-attn.

The real flash-attn kernels are never called when training scripts pass
`--attn_implementation sdpa`. This stub satisfies the import chain that
OpenRLHF performs unconditionally at module load, without requiring a
compiled flash-attn wheel.
"""

__version__ = "2.8.3-stub"


def _missing(*args, **kwargs):
    raise RuntimeError(
        "flash-attn stub invoked at runtime - use --attn_implementation sdpa "
        "or install the real flash-attn package."
    )


flash_attn_func = _missing
flash_attn_varlen_func = _missing
flash_attn_qkvpacked_func = _missing
flash_attn_kvpacked_func = _missing
flash_attn_varlen_qkvpacked_func = _missing
flash_attn_varlen_kvpacked_func = _missing
flash_attn_with_kvcache = _missing
