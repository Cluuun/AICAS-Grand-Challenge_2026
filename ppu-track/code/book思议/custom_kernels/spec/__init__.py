"""Speculative decoding: EAGLE draft, n-gram prompt-lookup, batch verify."""

from .decode import spec_decode_generate  # noqa: F401
from .ngram import ngram_spec_generate  # noqa: F401
