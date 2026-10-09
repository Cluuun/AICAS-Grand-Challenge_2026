"""Debug-only profiling helpers for AICAS runtime.

This module is intentionally dormant unless AICAS_DEBUG=1. The prefill profiler
uses CUDA events around coarse language-model modules, avoiding a synchronize per
decoder layer. It still perturbs timings slightly, so it is for diagnostics only.
"""

import os
import time

import torch

_PREFILL_PROFILE = None


def debug_enabled():
    return os.environ.get("AICAS_DEBUG", "0") != "0"


def debug_sync(device=None):
    if debug_enabled() and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def profile_enabled():
    return debug_enabled() and os.environ.get("AICAS_DEBUG_PREFILL_PROFILE", "1") != "0"


def _record(name, elapsed_ms):
    profile = globals().get("_PREFILL_PROFILE")
    if profile is None:
        return
    profile["times"][name] = profile["times"].get(name, 0.0) + float(elapsed_ms)


def _install_prefill_profiler(model):
    if not profile_enabled():
        return
    lm = getattr(getattr(model, "model", None), "language_model", None)
    if lm is None or getattr(lm, "_aicas_prefill_profiler_installed", False):
        return

    def wrap_module(module, name):
        if module is None or getattr(module, "_aicas_profile_wrapped", False):
            return
        orig_forward = module.forward

        def profiled_forward(*args, **kwargs):
            profile = globals().get("_PREFILL_PROFILE")
            if profile is None:
                return orig_forward(*args, **kwargs)
            if torch.cuda.is_available():
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                out = orig_forward(*args, **kwargs)
                end.record()
                profile["events"].append((name, start, end))
                return out
            t0 = time.perf_counter()
            out = orig_forward(*args, **kwargs)
            _record(name, (time.perf_counter() - t0) * 1000)
            return out

        module._aicas_profile_orig_forward = orig_forward
        module.forward = profiled_forward
        module._aicas_profile_wrapped = True

    wrap_module(getattr(lm, "embed_tokens", None), "embed_tokens")
    layers = getattr(lm, "layers", None)
    if layers is not None:
        for idx, layer in enumerate(layers):
            wrap_module(layer, f"layer{idx}")
    wrap_module(getattr(lm, "norm", None), "final_norm")
    lm._aicas_prefill_profiler_installed = True


def start_prefill_profile(model):
    if not profile_enabled():
        return None
    _install_prefill_profiler(model)
    globals()["_PREFILL_PROFILE"] = {"events": [], "times": {}}
    return globals()["_PREFILL_PROFILE"]


def stop_prefill_profile():
    profile = globals().get("_PREFILL_PROFILE")
    globals()["_PREFILL_PROFILE"] = None
    if not profile:
        return None
    events = profile.get("events") or []
    if events:
        torch.cuda.synchronize()
        for name, start, end in events:
            _record_to(profile, name, start.elapsed_time(end))
    return profile.get("times") or None


def _record_to(profile, name, elapsed_ms):
    profile["times"][name] = profile["times"].get(name, 0.0) + float(elapsed_ms)


def format_prefill_profile(profile):
    if not profile:
        return ""
    layer_items = []
    layer_sum = 0.0
    for key, value in profile.items():
        if key.startswith("layer"):
            try:
                idx = int(key[5:])
            except ValueError:
                idx = 9999
            layer_items.append((idx, key, value))
            layer_sum += value
    layer_items.sort()
    parts = []
    for key in ("embed_tokens", "final_norm"):
        if key in profile:
            parts.append(f"{key}_ms:{profile[key]:.3f}")
    parts.append(f"layer_sum_ms:{layer_sum:.3f}")
    if layer_items:
        slow = sorted(layer_items, key=lambda x: x[2], reverse=True)[:5]
        parts.append(
            "slow_layers:"
            + ",".join(f"{name}:{value:.3f}" for _, name, value in slow)
        )
        parts.append(
            "layer_ms:"
            + ",".join(f"{name}:{value:.3f}" for _, name, value in layer_items)
        )
    parts.append(f"profile_sum_ms:{sum(profile.values()):.3f}")
    return "|".join(parts)
