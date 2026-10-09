"""Vision encoder optimization and resolution routing."""

from .opt import patch_vision, warmup_vision_kernels  # noqa: F401
from .resolution_router import get_resolution_router  # noqa: F401
