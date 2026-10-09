from __future__ import annotations


def ensure_huggingface_hub_compat() -> None:
    try:
        import huggingface_hub
        from huggingface_hub import constants as hub_constants
    except Exception:
        return

    if not hasattr(huggingface_hub, "is_offline_mode"):
        def is_offline_mode() -> bool:
            try:
                return bool(getattr(hub_constants, "HF_HUB_OFFLINE", False))
            except Exception:
                return False

        huggingface_hub.is_offline_mode = is_offline_mode

