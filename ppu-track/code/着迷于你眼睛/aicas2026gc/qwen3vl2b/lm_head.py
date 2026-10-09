from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_ALLOWED_TOKENS_JSON = MODULE_DIR / "allowed_tokens.json"
DEFAULT_TOKEN_ID_MAP_PT = MODULE_DIR / "lm_head_token_id_map.pt"
RULE_VERSION = 1
BYTE_LEVEL_PREFIXES = ("\u0120", "\u010a", "\u0109")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_ascii_text_token(token: str) -> bool:
    idx = 0
    while idx < len(token) and token[idx] in BYTE_LEVEL_PREFIXES:
        idx += 1
    if idx == len(token):
        return idx > 0
    return all(0x20 <= ord(ch) <= 0x7E for ch in token[idx:])


def _read_full_vocab_size(config_path: Path | None, token_ids: list[int]) -> int:
    if config_path is not None and config_path.is_file():
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
        text_config = config.get("text_config", {})
        vocab_size = text_config.get("vocab_size", config.get("vocab_size"))
        if isinstance(vocab_size, int) and vocab_size > 0:
            return vocab_size
    return max(token_ids) + 1


def build_lm_head_token_artifacts(
    tokenizer_path: str | os.PathLike[str],
    out_json_path: str | os.PathLike[str] = DEFAULT_ALLOWED_TOKENS_JSON,
    out_pt_path: str | os.PathLike[str] = DEFAULT_TOKEN_ID_MAP_PT,
    config_path: str | os.PathLike[str] | None = None,
) -> torch.Tensor:
    tokenizer_path = Path(tokenizer_path)
    out_json_path = Path(out_json_path)
    out_pt_path = Path(out_pt_path)
    config_path = Path(config_path) if config_path is not None else tokenizer_path.with_name("config.json")

    with tokenizer_path.open("r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    records_by_id: dict[int, dict[str, Any]] = {}
    all_token_ids: list[int] = []
    vocab = tokenizer.get("model", {}).get("vocab", {})
    for token, token_id in vocab.items():
        if not isinstance(token_id, int):
            continue
        all_token_ids.append(token_id)
        if _is_ascii_text_token(token):
            records_by_id[token_id] = {"id": token_id, "token": token, "reason": "ascii_text"}

    for added_token in tokenizer.get("added_tokens", []):
        token_id = added_token.get("id")
        token = added_token.get("content")
        if not isinstance(token_id, int) or not isinstance(token, str):
            continue
        all_token_ids.append(token_id)
        reason = "qwen_special_token" if added_token.get("special") else "qwen_added_token"
        records_by_id[token_id] = {"id": token_id, "token": token, "reason": reason}

    full_vocab_size = _read_full_vocab_size(config_path, all_token_ids)
    records = [records_by_id[token_id] for token_id in sorted(records_by_id) if 0 <= token_id < full_vocab_size]
    token_id_map = torch.tensor([record["id"] for record in records], dtype=torch.int64)

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_pt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(token_id_map.cpu(), out_pt_path)

    reason_counts = Counter(record["reason"] for record in records)
    payload = {
        "metadata": {
            "rule_version": RULE_VERSION,
            "source_tokenizer": str(tokenizer_path),
            "source_tokenizer_sha256": _sha256(tokenizer_path),
            "source_config": str(config_path) if config_path is not None and config_path.is_file() else None,
            "full_vocab_size": full_vocab_size,
            "allowed_vocab_size": int(token_id_map.numel()),
            "reason_counts": dict(sorted(reason_counts.items())),
            "byte_level_prefixes": list(BYTE_LEVEL_PREFIXES),
        },
        "tokens": records,
    }
    with out_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"[LMHeadBuild] full_vocab={full_vocab_size} allowed_vocab={token_id_map.numel()} json={out_json_path} pt={out_pt_path}",
        flush=True,
    )
    return token_id_map


def _resolve_map_path(token_id_map_path: str | os.PathLike[str] | None = None) -> Path:
    env_path = os.environ.get("AICAS_LM_HEAD_TOKEN_ID_MAP", "").strip()
    if token_id_map_path is not None:
        return Path(token_id_map_path)
    if env_path:
        return Path(env_path)
    return DEFAULT_TOKEN_ID_MAP_PT


def _resolve_metadata_path(metadata_path: str | os.PathLike[str] | None = None) -> Path:
    env_path = os.environ.get("AICAS_LM_HEAD_ALLOWED_TOKENS_JSON", "").strip()
    if metadata_path is not None:
        return Path(metadata_path)
    if env_path:
        return Path(env_path)
    return DEFAULT_ALLOWED_TOKENS_JSON


def load_lm_head_token_id_map(
    token_id_map_path: str | os.PathLike[str] | None = None,
    *,
    device: torch.device | str | None = None,
    full_vocab_size: int | None = None,
) -> torch.Tensor:
    path = _resolve_map_path(token_id_map_path)
    if not path.is_file():
        raise FileNotFoundError(f"missing lm_head token id map: {path}")

    payload = torch.load(path, map_location="cpu")
    token_id_map = payload["token_id_map"] if isinstance(payload, dict) else payload
    if not isinstance(token_id_map, torch.Tensor):
        raise TypeError(f"{path} must contain a Tensor or a dict with token_id_map")
    if token_id_map.ndim != 1:
        raise ValueError(f"lm_head token id map must be 1D, got shape={tuple(token_id_map.shape)}")
    if token_id_map.dtype != torch.int64:
        token_id_map = token_id_map.to(torch.int64)
    if token_id_map.numel() == 0:
        raise ValueError("lm_head token id map is empty")
    if int(token_id_map.min().item()) < 0:
        raise ValueError("lm_head token id map contains negative token ids")
    if full_vocab_size is not None and int(token_id_map.max().item()) >= full_vocab_size:
        raise ValueError(f"lm_head token id map max={int(token_id_map.max().item())} exceeds full_vocab_size={full_vocab_size}")
    if torch.unique(token_id_map).numel() != token_id_map.numel():
        raise ValueError("lm_head token id map contains duplicate token ids")
    return token_id_map.to(device=device, non_blocking=True) if device is not None else token_id_map


def _load_metadata(metadata_path: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    path = _resolve_metadata_path(metadata_path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else None


def init_lm_head(
    lm_head: nn.Linear,
    norm_weight: torch.Tensor,
    *,
    token_id_map_path: str | os.PathLike[str] | None = None,
    metadata_path: str | os.PathLike[str] | None = None,
) -> torch.Tensor:
    full_vocab_size = int(lm_head.weight.shape[0])
    token_id_map = load_lm_head_token_id_map(
        token_id_map_path,
        device=lm_head.weight.device,
        full_vocab_size=full_vocab_size,
    )

    old_requires_grad = lm_head.weight.requires_grad
    lm_head_dtype = lm_head.weight.dtype
    pruned_weight = lm_head.weight.index_select(0, token_id_map)
    fused_weight = (pruned_weight.float() * norm_weight.float()).to(lm_head_dtype).contiguous()
    lm_head.weight = nn.Parameter(fused_weight, requires_grad=old_requires_grad)
    lm_head.out_features = int(token_id_map.numel())

    metadata = _load_metadata(metadata_path)
    metadata_vocab = metadata.get("allowed_vocab_size") if metadata is not None else None
    print(
        f"[LMHead] full_vocab={full_vocab_size} allowed_vocab={token_id_map.numel()} metadata_allowed_vocab={metadata_vocab} weight_shape={tuple(lm_head.weight.shape)}",
        flush=True,
    )
    return token_id_map


def lm_head_argmax(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    token_id_map: torch.Tensor | None = None,
) -> torch.Tensor:
    lm_hs = hidden_states[:, -1:, :].flatten()[None, None]
    local_tokens = torch.argmax(F.linear(lm_hs.view(1, -1), lm_head_weight), dim=-1)
    if token_id_map is None or token_id_map.numel() == 0:
        return local_tokens
    return token_id_map[local_tokens]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and inspect Qwen3VL lm_head token-map artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Generate allowed_tokens.json and lm_head_token_id_map.pt")
    build_parser.add_argument("--tokenizer", default="Qwen3-VL-2B-Instruct/tokenizer.json")
    build_parser.add_argument("--config", default="Qwen3-VL-2B-Instruct/config.json")
    build_parser.add_argument("--out-json", default=str(DEFAULT_ALLOWED_TOKENS_JSON))
    build_parser.add_argument("--out-pt", default=str(DEFAULT_TOKEN_ID_MAP_PT))

    args = parser.parse_args()
    if args.command == "build":
        build_lm_head_token_artifacts(
            tokenizer_path=args.tokenizer,
            out_json_path=args.out_json,
            out_pt_path=args.out_pt,
            config_path=args.config,
        )


if __name__ == "__main__":
    main()
