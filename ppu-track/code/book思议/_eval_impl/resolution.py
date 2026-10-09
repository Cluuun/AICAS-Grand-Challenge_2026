"""Resolution Router：图像分辨率路由与视觉 token 剪枝。"""

import os

from .config import _debug_print, get_resolution_router


_ROUTER_DEBUG_PRINTED = False


def _load_resolution_router():
    global _ROUTER_DEBUG_PRINTED
    if os.environ.get("AICAS_DISABLE_RESOLUTION_ROUTER", "0") == "1" or get_resolution_router is None:
        if not _ROUTER_DEBUG_PRINTED:
            _debug_print(
                "router",
                f"disabled_or_missing import_ok={get_resolution_router is not None} "
                f"disable_env={os.environ.get('AICAS_DISABLE_RESOLUTION_ROUTER')}",
                default_limit=20,
            )
            _ROUTER_DEBUG_PRINTED = True
        return None
    try:
        router = get_resolution_router()
        if not _ROUTER_DEBUG_PRINTED:
            _debug_print(
                "router",
                f"loaded={router is not None} type={getattr(router, 'model_type', None)} "
                f"path={getattr(router, 'path', None)} resolutions={getattr(router, 'resolutions', None)} "
                f"min_pixels={getattr(router, 'min_pixels', None)}",
                default_limit=20,
            )
            _ROUTER_DEBUG_PRINTED = True
        return router
    except Exception as e:
        if not _ROUTER_DEBUG_PRINTED:
            _debug_print("router", f"load_exception={type(e).__name__}: {e}", default_limit=20)
            _ROUTER_DEBUG_PRINTED = True
        return None


def _is_prune_sensitive(question):
    router = _load_resolution_router()
    if router is None:
        return True
    try:
        if hasattr(router, "predict_prune_sensitive"):
            return router.predict_prune_sensitive(question)
        return router.predict_high_res(question)
    except Exception:
        return True


def _needs_high_res_image(question, image_size=None):
    router = _load_resolution_router()
    if router is None:
        return True
    try:
        return router.predict_high_res(question, image_size)
    except Exception:
        return True


def _select_image_pixels(question, image_size=None):
    router = _load_resolution_router()
    if router is None or not hasattr(router, "predict_pixels"):
        return None
    try:
        pixels = router.predict_pixels(question, image_size)
        max_low_res = int(os.environ.get("AICAS_MAX_LOW_RES_PIXELS", "0"))
        if max_low_res > 0 and pixels is not None:
            pixels = min(pixels, max_low_res)
        return pixels
    except Exception:
        return None
