from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .conf import conf_bool, conf_int, conf_str

RouterType = str  # safe_short | cautious_short | general


@dataclass
class RouterDecision:
    router_type: RouterType = "general"
    prompt_added: int = 0
    prompt_text: str = ""
    question: str = ""
    router_rule: str = "default_general"
    original_question: str = ""


def _router_enabled() -> bool:
    return conf_bool("ENABLE_PROMPT_SHORT_ANSWER_ROUTER", "0")


def _legacy_global_enabled() -> bool:
    return conf_bool("ENABLE_PROMPT_SHORT_ANSWER", "0")


def _log_stats_enabled() -> bool:
    return conf_bool("PROMPT_SHORT_ANSWER_LOG_STATS", "0")


def safe_prompt_text() -> str:
    return conf_str("PROMPT_SHORT_ANSWER_SAFE_TEXT", "Answer only.").strip()


def cautious_prompt_text() -> str:
    return conf_str(
        "PROMPT_SHORT_ANSWER_CAUTIOUS_TEXT",
        "Answer only. If unclear, say unknown.",
    ).strip()


def cautious_prompt_enabled() -> bool:
    return conf_bool("PROMPT_SHORT_ANSWER_ENABLE_CAUTIOUS", "0")


def text_prefill_bucket_len() -> int:
    return conf_int("TEXT_PREFILL_CUDA_GRAPH_BUCKET_LEN", "125", minimum=1)


def _strip_benchmark_suffix(text: str) -> str:
    """TTFT 用 question+'.'、吞吐用 question+'。'；路由前去掉尾部句号差异。"""
    stripped = (text or "").strip()
    while stripped and stripped[-1] in (".", "。", "\u3002"):
        stripped = stripped[:-1].rstrip()
    return stripped


def _normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", _strip_benchmark_suffix(text).lower())


def _word_count(text: str) -> int:
    return len(re.findall(r"[a-z0-9']+", text))


def _contains_any(text: str, phrases: Tuple[str, ...]) -> bool:
    return any(p in text for p in phrases)


def _matches_pattern(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def extract_user_question(messages: List[Any]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            parts = [str(item.get("text", "")) for item in content if isinstance(item, dict) and "text" in item]
            return " ".join(p for p in parts if p).strip()
        if isinstance(content, str):
            return content.strip()
    return ""


def classify_question(question: str) -> RouterDecision:
    """生成前路由：cautious 优先于 safe，不确定则 general。"""
    raw = _strip_benchmark_suffix(question or "")
    text = _normalize_question(raw)
    decision = RouterDecision(original_question=raw, question=text)

    if not text:
        decision.router_rule = "empty_question"
        return decision

    if _word_count(text) >= 28 or text.count("?") >= 2:
        decision.router_rule = "long_or_multi_question"
        return decision

    if _matches_general(text):
        decision.router_rule = "general_intent"
        return decision

    cautious_rule = _match_cautious(text)
    if cautious_rule is not None:
        decision.router_type = "cautious_short"
        decision.router_rule = cautious_rule
        if cautious_prompt_enabled():
            decision.prompt_text = cautious_prompt_text()
            decision.prompt_added = 1 if decision.prompt_text else 0
        return decision

    safe_rule = _match_safe(text)
    if safe_rule is not None:
        decision.router_type = "safe_short"
        decision.router_rule = safe_rule
        decision.prompt_text = safe_prompt_text()
        decision.prompt_added = 1 if decision.prompt_text else 0
        return decision

    decision.router_rule = "default_general"
    return decision


def _matches_general(text: str) -> bool:
    general_phrases = (
        "describe",
        "explain",
        " why ",
        " why?",
        "what is happening",
        "what's happening",
        "what are happening",
        "what items",
        " list ",
        " list?",
        "compare",
        "difference between",
        "similarities",
        "step by step",
        "in detail",
        "tell me about",
        "tell me more",
    )
    if _contains_any(f" {text} ", tuple(general_phrases)):
        return True
    if _matches_pattern(text, r"\bhow\b") and not _matches_pattern(
        text, r"\bhow (many|much|often|long|far|old|man)\b"
    ):
        return True
    if _matches_pattern(text, r"\bwhat (are|were|do|does|did|can|could|should|would)\b"):
        if not _matches_pattern(
            text,
            r"\bwhat (color|colour|brand|letter|word|team|league|sport|type of)\b",
        ):
            return True
    return False


def _match_cautious(text: str) -> Optional[str]:
    rules: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
        ("cautious_price", ("how much", " price", " prices", "priced", "$", " cost", " costs", "for $")),
        ("cautious_time", (" what time", " time ", " date", " year", " when ", "what year", "what date")),
        (
            "cautious_percent_count",
            (
                "percent",
                "%",
                " how many",
                " how man ",
                " number of",
                " what number",
                " which number",
                " count",
                " price tags",
                " price tag",
                " measurement",
            ),
        ),
        ("cautious_ocr_read", (" read", " reads", " reading", " says", " said", " sign says", " written", " text on")),
        ("cautious_value", (" worth", " value", " valued")),
        (
            "cautious_absence",
            (
                "not visible",
                "can't see",
                "cannot see",
                "can you determine",
                "can't determine",
                "cannot determine",
                "is there any",
                "are there any",
                "unable to",
            ),
        ),
    )
    padded = f" {text} "
    for rule, phrases in rules:
        if _contains_any(padded, phrases):
            return rule
    if _matches_pattern(text, r"\bhow many\b"):
        return "cautious_how_many"
    if _matches_pattern(text, r"\bwhat year\b"):
        return "cautious_year"
    return None


def _match_safe(text: str) -> Optional[str]:
    padded = f" {text} "
    yes_no = (
        "is this ",
        "is there ",
        "is the ",
        "is it ",
        "are these ",
        "are there ",
        "are the ",
        "does this ",
        "does the ",
        "do you see",
        "can you see",
        "do you see",
    )
    if _contains_any(padded, ("on or off", "turned on", "turned off", " switches", " switch ", "switch on", "switch off")):
        return "safe_on_off"

    if _contains_any(padded, ("what color", "what colour", "color of", "colour of", "letter color", "letters color")):
        return "safe_color"

    if _contains_any(
        padded,
        (
            "what brand",
            "which brand",
            "brand advertised",
            "brands being advertised",
            "brands advertised",
            "being advertised",
            "brand of",
            " label brand",
            "logo on",
            "logo says",
            "what logo",
            "which logo",
            "sponsor",
            "sponsors",
        ),
    ):
        return "safe_brand"

    if _contains_any(
        padded,
        ("what letter", "which letter", "what word", "which word", "word in", "letter on", "letters on"),
    ):
        return "safe_word_letter"

    if _contains_any(padded, yes_no) or _matches_pattern(text, r"^(is|are|does|do|can)\b"):
        return "safe_yes_no"

    if _contains_any(
        padded,
        (
            "left or right",
            "right or left",
            "top or bottom",
            "bottom or top",
            "which side",
            "on the left",
            "on the right",
            "farthest left",
            "farthest right",
            "far left",
            "far right",
        ),
    ):
        return "safe_direction"

    if _matches_pattern(text, r"\bis this\b") or _matches_pattern(text, r"\bis that\b"):
        return "safe_is_this"

    return None


def _append_instruction_to_user_text(messages: List[Any], instruction: str) -> List[Any]:
    """浅拷贝消息，保留 PIL image 对象引用，避免影响 vision cache key。"""
    out: List[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            out.append(message)
            continue
        msg = dict(message)
        content = msg.get("content")
        if isinstance(content, list):
            new_content: List[Any] = []
            appended = False
            for item in content:
                if isinstance(item, dict) and "text" in item and not appended:
                    new_item = dict(item)
                    base = str(new_item["text"]).rstrip()
                    new_item["text"] = f"{base}\n{instruction}" if base else instruction
                    new_content.append(new_item)
                    appended = True
                else:
                    new_content.append(item)
            msg["content"] = new_content
        elif isinstance(content, str):
            base = content.rstrip()
            msg["content"] = f"{base}\n{instruction}" if base else instruction
        out.append(msg)
    return out


def _legacy_instruction() -> str:
    return conf_str("PROMPT_SHORT_ANSWER_TEXT", "Brief answer only.").strip()


def _legacy_mode() -> str:
    return conf_str("PROMPT_SHORT_ANSWER_MODE", "user_suffix", lower=True)


def _suffix_rendered_chat(text: str, instruction: str) -> str:
    if not instruction:
        return text
    marker = "<|im_start|>assistant"
    idx = text.rfind(marker)
    if idx >= 0:
        prefix = text[:idx].rstrip()
        suffix = text[idx:]
        return f"{prefix}\n{instruction}\n{suffix}"
    return f"{text.rstrip()}\n{instruction}\n"


def _init_stats(processor: Any) -> Dict[str, Any]:
    stats = getattr(processor, "_prompt_short_answer_stats", None)
    if not isinstance(stats, dict):
        stats = {
            "safe_short": 0,
            "cautious_short": 0,
            "general": 0,
            "prompt_added": 0,
            "prompt_len_over_bucket": 0,
            "calls": 0,
        }
        processor._prompt_short_answer_stats = stats
    return stats


def _record_decision(processor: Any, decision: RouterDecision, *, new_prompt_len: Optional[int] = None) -> None:
    if processor is None:
        return
    bucket = text_prefill_bucket_len()
    last = {
        "router_type": decision.router_type,
        "prompt_added": int(decision.prompt_added),
        "prompt_text": decision.prompt_text,
        "question": decision.original_question,
        "router_rule": decision.router_rule,
        "original_prompt_len": getattr(processor, "_prompt_short_answer_original_len", None),
        "new_prompt_len": new_prompt_len,
        "prompt_len_over_bucket": int(new_prompt_len is not None and new_prompt_len > bucket),
    }
    processor._prompt_short_answer_last = last

    stats = _init_stats(processor)
    stats["calls"] = int(stats.get("calls", 0)) + 1
    stats[decision.router_type] = int(stats.get(decision.router_type, 0)) + 1
    if decision.prompt_added:
        stats["prompt_added"] = int(stats.get("prompt_added", 0)) + 1
    if new_prompt_len is not None and new_prompt_len > bucket:
        stats["prompt_len_over_bucket"] = int(stats.get("prompt_len_over_bucket", 0)) + 1

    if _log_stats_enabled():
        print(
            "[prompt_short_answer] "
            f"type={decision.router_type} rule={decision.router_rule} "
            f"added={decision.prompt_added} prompt={decision.prompt_text!r} "
            f"len={new_prompt_len} over_bucket={last['prompt_len_over_bucket']}"
        )


def route_question(question: str) -> RouterDecision:
    return classify_question(question)


def prepare_messages_for_render(messages: List[Any], *, processor: Any = None) -> List[Any]:
    if not _router_enabled() and not _legacy_global_enabled():
        return messages

    question = extract_user_question(messages)
    if _router_enabled():
        decision = classify_question(question)
        _record_decision(processor, decision)
        if decision.prompt_added and decision.prompt_text:
            return _append_instruction_to_user_text(messages, decision.prompt_text)
        return messages

    if _legacy_global_enabled():
        instruction = _legacy_instruction()
        if instruction:
            legacy = RouterDecision(
                router_type="legacy_global",
                prompt_added=1,
                prompt_text=instruction,
                question=question,
                original_question=question,
                router_rule="legacy_enable_prompt_short_answer",
            )
            _record_decision(processor, legacy)
            if _legacy_mode() == "user_suffix":
                return _append_instruction_to_user_text(messages, instruction)
        return messages

    _record_decision(processor, RouterDecision(original_question=question, question=_normalize_question(question)))
    return messages


def finish_rendered_prompt(text: str) -> str:
    if _router_enabled() or _legacy_global_enabled():
        return text
    instruction = _legacy_instruction()
    if not instruction:
        return text
    if _legacy_mode() == "render_suffix":
        return _suffix_rendered_chat(text, instruction)
    return text


def finalize_prompt_len(processor: Any, prompt_len: int) -> None:
    last = getattr(processor, "_prompt_short_answer_last", None)
    if isinstance(last, dict):
        last["new_prompt_len"] = int(prompt_len)
        bucket = text_prefill_bucket_len()
        last["prompt_len_over_bucket"] = int(prompt_len > bucket)
        stats = _init_stats(processor)
        if prompt_len > bucket:
            stats["prompt_len_over_bucket"] = int(stats.get("prompt_len_over_bucket", 0)) + 1
