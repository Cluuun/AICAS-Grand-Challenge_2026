"""Tiny CPU-side question router for dynamic image resolution.

The router is intentionally independent from torch/transformers. Runtime work is
limited to token/character feature extraction and a small linear model, so it is
cheap enough to run before image preprocessing.
"""

import math
import os
import re
import pickle
import zlib


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(question):
    return _TOKEN_RE.findall((question or "").lower())


def _feature_names(question, image_size=None):
    toks = _tokens(question)
    yield "__bias__"
    for tok in toks:
        yield f"w:{tok}"
        if tok.isdigit():
            yield "has_digit_token"
    for a, b in zip(toks, toks[1:]):
        yield f"b:{a}_{b}"

    compact = " ".join(toks)
    for n in (3, 4):
        if len(compact) >= n:
            for i in range(len(compact) - n + 1):
                yield f"c{n}:{compact[i:i + n]}"

    if image_size is not None:
        try:
            w, h = image_size
            area = int(w) * int(h)
            if area >= 1_000_000:
                yield "img_area:large"
            if max(w, h) >= 1200:
                yield "img_edge:large"
            if min(w, h) > 0 and max(w, h) / min(w, h) >= 1.8:
                yield "img_aspect:wide"
        except Exception:
            pass


def _feature_index(name, dim):
    return zlib.crc32(name.encode("utf-8")) % dim


class QuestionResolutionRouter:
    """Hashing linear router.

    Model format:
    {
      "version": 1,
      "dim": 512,
      "threshold": 0.5,
      "weights": [...],
      "bias": 0.0
    }
    Old binary models predict whether to keep original high resolution. New
    ordinal models predict the lowest fixed resolution rank to use.
    """

    def __init__(self, data):
        self.model_type = data.get("type", "hashed_logistic_question_resolution_router")
        self.threshold = float(data.get("threshold", 0.5))
        self.min_pixels = int(data.get("min_pixels", os.environ.get("AICAS_MIN_ROUTER_PIXELS", "49152")))
        if self.model_type == "lexical_resolution_router":
            self.resolutions = [str(x) for x in data.get("resolutions", ["262144", "original"])]
            self.low_resolution = str(data.get("low_resolution", self.resolutions[0]))
            self.high_resolution = str(data.get("high_resolution", self.resolutions[-1]))
            self.always_high_res = bool(data.get("always_high_res", False))
            high_phrases = data.get("high_res_phrases", data.get("phrases", []))
            high_regexes = data.get("high_res_regexes", data.get("regexes", []))
            prune_phrases = data.get("prune_sensitive_phrases", high_phrases)
            prune_regexes = data.get("prune_sensitive_regexes", high_regexes)
            self.high_res_phrases = [str(x).lower() for x in high_phrases]
            self.high_res_regexes = [re.compile(str(x), re.IGNORECASE) for x in high_regexes]
            self.prune_sensitive_phrases = [str(x).lower() for x in prune_phrases]
            self.prune_sensitive_regexes = [re.compile(str(x), re.IGNORECASE) for x in prune_regexes]
            self.dim = 0
        elif self.model_type == "hashed_ordinal_resolution_router":
            self.resolutions = [str(x) for x in data["resolutions"]]
            self.thresholds = [float(x) for x in data.get("thresholds", [])]
            self.biases = [float(x) for x in data["biases"]]
            self.weights = [[float(v) for v in row] for row in data["weights"]]
            self.dim = int(data.get("dim") or (len(self.weights[0]) if self.weights else 0))
            self.conservative_steps = int(data.get("conservative_steps", 0))
        else:
            self.resolutions = ["262144", "original"]
            self.bias = float(data.get("bias", 0.0))
            self.weights = [float(x) for x in data["weights"]]
            self.dim = len(self.weights)

    @classmethod
    def from_file(cls, path):
        with open(path, "rb") as f:
            raw = f.read()
        try:
            data = pickle.loads(raw)
        except Exception:
            import json

            data = json.loads(raw.decode("utf-8"))
        return cls(data)

    def _score(self, weights, bias, question, image_size=None):
        if self.dim <= 0:
            return float(bias)
        s = float(bias)
        for name in _feature_names(question, image_size):
            s += weights[_feature_index(name, self.dim)]
        return s

    def score(self, question, image_size=None):
        if self.model_type == "hashed_ordinal_resolution_router":
            if not self.weights:
                return 0.0
            return self._score(self.weights[-1], self.biases[-1], question, image_size)
        return self._score(self.weights, self.bias, question, image_size)

    def predict_high_res(self, question, image_size=None):
        if self.model_type == "lexical_resolution_router":
            return self.predict_resolution(question, image_size) == self.high_resolution
        if self.model_type == "hashed_ordinal_resolution_router":
            return self.predict_resolution(question, image_size) == "original"
        # Compare logits directly to avoid exp() in the hot path when possible.
        if self.threshold <= 0.0:
            return True
        if self.threshold >= 1.0:
            return False
        logit_threshold = math.log(self.threshold / (1.0 - self.threshold))
        return self.score(question, image_size) >= logit_threshold

    def predict_resolution(self, question, image_size=None):
        if self.model_type == "lexical_resolution_router":
            if self.always_high_res:
                return self.high_resolution
            q = (question or "").lower()
            if any(phrase in q for phrase in self.high_res_phrases):
                return self.high_resolution
            if any(regex.search(question or "") for regex in self.high_res_regexes):
                return self.high_resolution
            return self.low_resolution
        if self.model_type != "hashed_ordinal_resolution_router":
            return "original" if self.predict_high_res(question, image_size) else self.resolutions[0]
        rank = 0
        for cutoff, (weights, bias) in enumerate(zip(self.weights, self.biases), start=1):
            threshold = self.thresholds[cutoff - 1] if cutoff - 1 < len(self.thresholds) else self.threshold
            if threshold <= 0.0:
                needs_at_least_cutoff = True
            elif threshold >= 1.0:
                needs_at_least_cutoff = False
            else:
                logit_threshold = math.log(threshold / (1.0 - threshold))
                needs_at_least_cutoff = self._score(weights, bias, question, image_size) >= logit_threshold
            if needs_at_least_cutoff:
                rank = cutoff
        bump = self.conservative_steps
        try:
            bump += int(os.environ.get("AICAS_RESOLUTION_ROUTER_BUMP", "0"))
        except Exception:
            pass
        rank = min(len(self.resolutions) - 1, max(0, rank + bump))
        return self.resolutions[rank]

    def predict_pixels(self, question, image_size=None):
        resolution = self.predict_resolution(question, image_size)
        if str(resolution).lower() == "original":
            return None
        try:
            pixels = int(resolution)
        except Exception:
            return None
        try:
            min_pixels = int(os.environ.get("AICAS_MIN_ROUTER_PIXELS", str(self.min_pixels)))
        except Exception:
            min_pixels = self.min_pixels
        return max(pixels, min_pixels)

    def predict_prune_sensitive(self, question, image_size=None):
        if self.model_type == "lexical_resolution_router":
            q = (question or "").lower()
            if any(phrase in q for phrase in self.prune_sensitive_phrases):
                return True
            return any(regex.search(question or "") for regex in self.prune_sensitive_regexes)
        return self.predict_high_res(question, image_size)


_ROUTER = None
_ROUTER_PATH = None
_ROUTER_MISSING = False


def get_resolution_router():
    global _ROUTER, _ROUTER_PATH, _ROUTER_MISSING
    path = os.environ.get("AICAS_RESOLUTION_ROUTER_MODEL", "")
    if not path:
        # custom_kernels/vision/ → custom_kernels/ → project root
        base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "model")
        path = os.path.join(base, "router_model.bin")
    if _ROUTER is not None and _ROUTER_PATH == path:
        return _ROUTER
    if _ROUTER_MISSING and _ROUTER_PATH == path:
        return None
    try:
        _ROUTER = QuestionResolutionRouter.from_file(path)
        _ROUTER_PATH = path
        _ROUTER_MISSING = False
        return _ROUTER
    except OSError:
        _ROUTER = None
        _ROUTER_PATH = path
        _ROUTER_MISSING = True
        return None
    except Exception:
        _ROUTER = None
        _ROUTER_PATH = path
        _ROUTER_MISSING = True
        return None
