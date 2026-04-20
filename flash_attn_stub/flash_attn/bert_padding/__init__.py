"""Stub for flash_attn.bert_padding.

These helpers are only called when `--packing_samples` is enabled in the
OpenRLHF trainer; for the default (non-packed) code path the imports are
satisfied but never invoked.
"""


def index_first_axis(input, indices):
    return input[indices]


def pad_input(hidden_states, indices, batch, seqlen):
    raise RuntimeError(
        "flash_attn.bert_padding.pad_input stub - only used with --packing_samples"
    )


def unpad_input(hidden_states, attention_mask):
    raise RuntimeError(
        "flash_attn.bert_padding.unpad_input stub - only used with --packing_samples"
    )
