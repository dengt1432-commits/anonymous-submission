"""Shared utilities for recording prefill and total inference timing in lmms-eval."""
import time
from typing import List

import torch


def is_prefill_forward(args, kwargs) -> bool:
    """Detect if this forward call is prefill (first pass, no cache)."""
    past_kv = kwargs.get("past_key_values", None)
    if past_kv is None:
        return True
    if hasattr(past_kv, "get_seq_length") and past_kv.get_seq_length() == 0:
        return True
    return False


def setup_prefill_timing_hook(model, prefill_times: List[float]):
    """Register forward hooks to record prefill phase time. Returns remove handles callable."""
    prefill_start = [None]  # use list to allow closure assignment

    def pre_hook(module, args, kwargs):
        if is_prefill_forward(args, kwargs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            prefill_start[0] = time.perf_counter()

    def post_hook(module, args, kwargs, output):
        if prefill_start[0] is not None:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - prefill_start[0]
            prefill_times.append(elapsed)
            prefill_start[0] = None

    raw = model
    while hasattr(raw, "module"):
        raw = raw.module
    handle_pre = raw.register_forward_pre_hook(pre_hook, with_kwargs=True)
    handle_post = raw.register_forward_hook(post_hook, with_kwargs=True)

    def remove():
        handle_pre.remove()
        handle_post.remove()

    return remove
