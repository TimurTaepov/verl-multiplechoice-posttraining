from __future__ import annotations

import re
from dataclasses import dataclass

from verl.utils.reward_score import gsm8k as verl_gsm8k


_GSM8K_ANSWER_RE = re.compile(r"####\s*([^\n]+)")
_FINAL_ANSWER_RE = re.compile(r"FINAL_ANSWER:\s*(.+)", re.IGNORECASE)
_REASONING_BLOCK_RE = re.compile(r"REASONING:\s*(.*?)(?:\n\s*FINAL_ANSWER:|\Z)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class ParsedAnswer:
    extracted_answer: str | None
    normalized_answer: str | None
    reason: str


def sanitize_answer_surface(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    text = text.strip(" \t\r\n'\"`")
    text = re.sub(r"\s+", " ", text)
    return text.strip() or None


def normalize_numeric_text(value: str | None) -> str | None:
    return verl_gsm8k._normalize_numeric_token(value)


def values_equal(left: str | None, right: str | None) -> bool:
    return verl_gsm8k._numeric_tokens_equal(left, normalize_numeric_text(right))


def parse_gold_answer(answer_text: str) -> ParsedAnswer:
    extracted = verl_gsm8k.extract_solution(answer_text, method="strict")
    if extracted is None:
        return ParsedAnswer(extracted_answer=None, normalized_answer=None, reason="missing_hash_answer")
    return ParsedAnswer(
        extracted_answer=extracted,
        normalized_answer=normalize_numeric_text(extracted),
        reason="hash_extraction",
    )


def format_gsm8k_cot_answer(answer_text: str) -> str:
    parsed = parse_gold_answer(answer_text)
    final_answer = sanitize_answer_surface(parsed.extracted_answer)
    reasoning = _GSM8K_ANSWER_RE.sub("", answer_text).strip()
    reasoning = re.sub(r"\n{3,}", "\n\n", reasoning)
    if final_answer is None:
        return answer_text.strip()
    if reasoning:
        return f"{reasoning}\nFINAL_ANSWER: {final_answer}"
    return f"FINAL_ANSWER: {final_answer}"


def extract_generated_answer(completion: str) -> ParsedAnswer:
    extracted = verl_gsm8k.extract_solution(completion, method="flexible")
    if extracted is None:
        return ParsedAnswer(extracted_answer=None, normalized_answer=None, reason="missing_final_answer")

    if _FINAL_ANSWER_RE.search(completion):
        reason = "final_answer"
    elif _GSM8K_ANSWER_RE.search(completion):
        reason = "hash_answer_fallback"
    else:
        reason = "flexible_numeric_fallback"

    return ParsedAnswer(
        extracted_answer=extracted,
        normalized_answer=normalize_numeric_text(extracted),
        reason=reason,
    )


def extract_reasoning_block(completion: str) -> str | None:
    match = _REASONING_BLOCK_RE.search(completion)
    if not match:
        return None
    return sanitize_answer_surface(match.group(1))
