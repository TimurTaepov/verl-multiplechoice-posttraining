from __future__ import annotations

import hashlib
import html
import json
import random
from pathlib import Path
from typing import Any

import datasets
import numpy as np

from reliable_gsm8k.parsing import extract_generated_answer, parse_gold_answer, values_equal
from reliable_gsm8k.prompts import build_mc_onecorrect_prompt
from verl import DataProto
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.reward_score import gsm8k_mc


STAGE1_SOURCE = "gsm8k_dynamic_mc_stage1"
STAGE2_SOURCE = "gsm8k_dynamic_mc_stage2"
MC_LABELS = ("A", "B", "C", "D")


def _get_int_config(config: Any, key: str, default: int, *, minimum: int) -> int:
    value = int(config.get(key, default))
    if value < minimum:
        raise ValueError(f"data.dynamic_mc.{key} must be >= {minimum}, got {value}.")
    return value


def _get_bool_config(config: Any, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    tmp_path.replace(path)


def question_id_from_question(question: str) -> str:
    return hashlib.md5(question.encode("utf-8")).hexdigest()


def build_stage1_correct_prompt(question: str) -> str:
    return (
        "Solve the GSM8K math problem step by step.\n"
        "Return exactly this format:\n"
        "REASONING: <step-by-step solution>\n"
        "FINAL_ANSWER: <final numeric answer>\n"
        "Use Arabic numerals in FINAL_ANSWER.\n\n"
        f"Question:\n{question.strip()}"
    )


def build_stage1_candidate_prompt(question: str) -> str:
    return (
        "Solve the GSM8K math problem step by step.\n"
        "Return exactly this format:\n"
        "REASONING: <step-by-step solution>\n"
        "FINAL_ANSWER: <final numeric answer>\n"
        "Use Arabic numerals in FINAL_ANSWER.\n\n"
        f"Question:\n{question.strip()}"
    )


def build_stage1_incorrect_prompt(question: str) -> str:
    return (
        "Write a plausible but mathematically incorrect step-by-step solution to the GSM8K problem.\n"
        "The reasoning should look like a realistic student mistake.\n"
        "The FINAL_ANSWER must be numeric and must be wrong.\n"
        "Return exactly this format:\n"
        "REASONING: <step-by-step but incorrect solution>\n"
        "FINAL_ANSWER: <wrong numeric answer>\n"
        "Use Arabic numerals in FINAL_ANSWER.\n\n"
        f"Question:\n{question.strip()}"
    )


def make_stage1_record(
    *,
    item_id: str,
    question: str,
    gold_answer: str,
    role_requested: str,
    slot: int,
) -> dict[str, Any]:
    question_id = question_id_from_question(question)
    if role_requested == "candidate":
        prompt = build_stage1_candidate_prompt(question)
    elif role_requested == "correct":
        prompt = build_stage1_correct_prompt(question)
    elif role_requested == "incorrect":
        prompt = build_stage1_incorrect_prompt(question)
    else:
        raise ValueError("role_requested must be 'candidate', 'correct', or 'incorrect'.")
    return {
        "data_source": STAGE1_SOURCE,
        "prompt": [{"role": "user", "content": prompt}],
        "question_id": question_id,
        "reward_model": {"style": "rule", "ground_truth": gold_answer},
        "extra_info": {
            "stage": "stage1_candidate",
            "item_id": item_id,
            "question_id": question_id,
            "question": question,
            "gold_answer": gold_answer,
            "role_requested": role_requested,
            "candidate_slot": slot,
            "correct_choice": "",
            "option_roles": {label: "" for label in MC_LABELS},
            "option_final_answers": {label: "" for label in MC_LABELS},
        },
    }


def make_stage1_records_for_question(
    *,
    item_id: str,
    question: str,
    gold_answer: str,
    incorrect_target_count: int = 3,
    prompt_mode: str = "neutral",
) -> list[dict[str, Any]]:
    if prompt_mode == "neutral":
        prompt_count = max(1, incorrect_target_count + 1)
        return [
            make_stage1_record(
                item_id=item_id,
                question=question,
                gold_answer=gold_answer,
                role_requested="candidate",
                slot=slot,
            )
            for slot in range(prompt_count)
        ]
    if prompt_mode != "role":
        raise ValueError("prompt_mode must be 'neutral' or 'role'.")

    records = [
        make_stage1_record(
            item_id=item_id,
            question=question,
            gold_answer=gold_answer,
            role_requested="correct",
            slot=0,
        )
    ]
    for slot in range(incorrect_target_count):
        records.append(
            make_stage1_record(
                item_id=item_id,
                question=question,
                gold_answer=gold_answer,
                role_requested="incorrect",
                slot=slot,
            )
        )
    return records


def make_stage1_records_from_gsm8k_example(
    *,
    split: str,
    index: int,
    example: dict[str, Any],
    incorrect_target_count: int = 3,
    prompt_mode: str = "neutral",
) -> list[dict[str, Any]]:
    question = str(example["question"])
    parsed = parse_gold_answer(str(example["answer"]))
    if parsed.normalized_answer is None:
        return []
    return make_stage1_records_for_question(
        item_id=f"gsm8k:{split}:{index}",
        question=question,
        gold_answer=parsed.normalized_answer,
        incorrect_target_count=incorrect_target_count,
        prompt_mode=prompt_mode,
    )


class VerifiedCandidate:
    __slots__ = ("completion", "final_answer", "role")

    def __init__(self, completion: str, final_answer: str, role: str) -> None:
        self.completion = completion
        self.final_answer = final_answer
        self.role = role


class GSM8KDynamicMCDataset(RLHFDataset):
    """RLHFDataset that turns Stage 1 rollouts into Stage 2 MC prompts inside VERL.

    Initial training rows are Stage 1 candidate-generation prompts. After each VERL
    training batch, `on_batch_end` parses the actor's generated candidate solutions,
    verifies them against the GSM8K gold answer, and appends a Stage 2 MC prompt when
    a question has one correct and three distinct incorrect candidate CoTs.
    """

    supports_trainer_resume = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        dynamic_cfg = self.config.get("dynamic_mc", {})
        self.incorrect_target_count = _get_int_config(dynamic_cfg, "stage2_incorrect_count", 3, minimum=0)
        if self.incorrect_target_count != 3:
            raise ValueError("data.dynamic_mc.stage2_incorrect_count must be 3 for four-option MC prompts.")
        self.max_stage2_per_question = _get_int_config(dynamic_cfg, "max_stage2_per_question", 1, minimum=1)
        self.max_new_stage2_per_batch = _get_int_config(dynamic_cfg, "max_new_stage2_per_batch", 256, minimum=1)
        self.stage2_candidate_max_chars = _get_int_config(dynamic_cfg, "stage2_candidate_max_chars", 2000, minimum=0)
        self.stage2_insert_strategy = str(dynamic_cfg.get("stage2_insert_strategy", "prepend"))
        if self.stage2_insert_strategy not in {"prepend", "append"}:
            raise ValueError("data.dynamic_mc.stage2_insert_strategy must be 'prepend' or 'append'.")
        self.seed = int(dynamic_cfg.get("seed", self.config.get("seed", 7) or 7))
        artifact_dir_raw = str(dynamic_cfg.get("artifact_dir", "")).strip()
        self.artifact_dir = Path(artifact_dir_raw).expanduser() if artifact_dir_raw else None
        self.artifact_include_completions = _get_bool_config(dynamic_cfg, "artifact_include_completions", True)
        self.coverage_chart_interval = _get_int_config(dynamic_cfg, "coverage_chart_interval", 100, minimum=0)
        self.coverage_chart_on_epoch = _get_bool_config(dynamic_cfg, "coverage_chart_on_epoch", True)
        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._candidate_buffer: dict[str, dict[str, Any]] = {}
        self._stage2_counts: dict[str, int] = {}
        self._hook_calls = 0
        self._stage1_seen_total = 0
        self._stage1_rejected_total = 0
        self._stage1_duplicate_total = 0
        self._accepted_correct_total = 0
        self._accepted_incorrect_total = 0
        self._stage2_queued_total = 0
        self._inserted_stage2_total = 0
        self._stage2_question_ids: set[str] = set()
        self._initial_question_count = self._count_initial_questions()
        self._pending_stage2_records: list[dict[str, Any]] = []
        self._coverage_history: list[dict[str, Any]] = []
        self._write_summary()
        self._record_coverage_history(event="init")

    def _count_initial_questions(self) -> int:
        try:
            return len({str(qid) for qid in self.dataframe["question_id"] if str(qid)})
        except Exception:
            question_ids: set[str] = set()
            for record in self.dataframe:
                question_id = str(record.get("question_id", ""))
                if question_id:
                    question_ids.add(question_id)
            return len(question_ids)

    def _artifact_path(self, name: str) -> Path | None:
        if self.artifact_dir is None:
            return None
        return self.artifact_dir / name

    def _append_artifact(self, name: str, payload: dict[str, Any]) -> None:
        path = self._artifact_path(name)
        if path is None:
            return
        _append_jsonl(path, payload)

    def _summary_payload(self) -> dict[str, Any]:
        questions_with_correct = sum(1 for entry in self._candidate_buffer.values() if entry["correct"])
        questions_with_incorrect = sum(1 for entry in self._candidate_buffer.values() if entry["incorrect"])
        successful_questions = len(self._stage2_question_ids)
        return {
            "initial_question_count": self._initial_question_count,
            "hook_calls": self._hook_calls,
            "stage1_seen_total": self._stage1_seen_total,
            "stage1_rejected_total": self._stage1_rejected_total,
            "stage1_duplicate_total": self._stage1_duplicate_total,
            "stage1_accepted_correct_total": self._accepted_correct_total,
            "stage1_accepted_incorrect_total": self._accepted_incorrect_total,
            "stage1_accepted_total": self._accepted_correct_total + self._accepted_incorrect_total,
            "questions_with_correct_candidate": questions_with_correct,
            "questions_with_incorrect_candidate": questions_with_incorrect,
            "candidate_buffer_question_count": len(self._candidate_buffer),
            "stage2_queued_total": self._stage2_queued_total,
            "stage2_inserted_total": self._inserted_stage2_total,
            "stage2_pending_total": len(self._pending_stage2_records),
            "successful_stage2_question_count": successful_questions,
            "successful_stage2_question_coverage": (
                successful_questions / self._initial_question_count if self._initial_question_count else 0.0
            ),
        }

    def _write_summary(self) -> None:
        path = self._artifact_path("dynamic_mc_summary.json")
        if path is None:
            return
        _write_json(path, self._summary_payload())

    def _coverage_payload(self, *, event: str, epoch: int | None = None) -> dict[str, Any]:
        payload = self._summary_payload()
        return {
            "event": event,
            "step": self._hook_calls,
            "epoch": epoch,
            "coverage": payload["successful_stage2_question_coverage"],
            "successful_stage2_question_count": payload["successful_stage2_question_count"],
            "initial_question_count": payload["initial_question_count"],
            "stage2_queued_total": payload["stage2_queued_total"],
            "stage2_inserted_total": payload["stage2_inserted_total"],
            "stage2_pending_total": payload["stage2_pending_total"],
            "stage1_seen_total": payload["stage1_seen_total"],
            "stage1_accepted_correct_total": payload["stage1_accepted_correct_total"],
            "stage1_accepted_incorrect_total": payload["stage1_accepted_incorrect_total"],
        }

    def _record_coverage_history(self, *, event: str, epoch: int | None = None) -> dict[str, Any]:
        payload = self._coverage_payload(event=event, epoch=epoch)
        self._coverage_history.append(payload)
        self._append_artifact("coverage_history.jsonl", payload)
        return payload

    def _write_coverage_chart(self, *, reason: str, epoch: int | None = None) -> None:
        if self.artifact_dir is None or not self._coverage_history:
            return

        width = 980
        height = 560
        left = 90
        right = 40
        top = 80
        bottom = 85
        chart_w = width - left - right
        chart_h = height - top - bottom
        max_step = max(1, max(int(point["step"]) for point in self._coverage_history))

        def x_for(step: int) -> float:
            return left + (step / max_step) * chart_w

        def y_for(coverage: float) -> float:
            return top + (1.0 - max(0.0, min(1.0, coverage))) * chart_h

        points = [
            f'{x_for(int(point["step"])):.2f},{y_for(float(point["coverage"])):.2f}'
            for point in self._coverage_history
        ]
        latest = self._coverage_history[-1]
        latest_pct = float(latest["coverage"]) * 100.0
        latest_count = int(latest["successful_stage2_question_count"])
        total_count = int(latest["initial_question_count"])

        grid_lines = []
        for pct in (0, 25, 50, 75, 100):
            y = y_for(pct / 100.0)
            grid_lines.append(
                f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" '
                'stroke="#E5E7EB" stroke-width="1" />'
            )
            grid_lines.append(
                f'<text x="{left - 14}" y="{y + 4:.2f}" text-anchor="end" '
                'font-family="Arial" font-size="12" fill="#4B5563">'
                f'{pct}%</text>'
            )

        event_markers = []
        for point in self._coverage_history:
            if str(point.get("event")) == "epoch_end":
                x = x_for(int(point["step"]))
                event_markers.append(
                    f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}" '
                    'stroke="#F59E0B" stroke-width="1.5" stroke-dasharray="6 5" />'
                )

        title = "Dynamic MC Stage 2 Coverage"
        subtitle = (
            f'latest: {latest_pct:.2f}% ({latest_count}/{total_count} questions), '
            f'step={latest["step"]}, reason={reason}'
        )
        if epoch is not None:
            subtitle += f', epoch={epoch}'

        svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#FFFFFF" />
  <text x="{left}" y="35" font-family="Arial" font-size="24" font-weight="700" fill="#111827">{html.escape(title)}</text>
  <text x="{left}" y="60" font-family="Arial" font-size="13" fill="#374151">{html.escape(subtitle)}</text>
  {''.join(grid_lines)}
  {''.join(event_markers)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" stroke-width="1.5" />
  <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" stroke-width="1.5" />
  <polyline points="{' '.join(points)}" fill="none" stroke="#2563EB" stroke-width="3" />
  <circle cx="{x_for(int(latest['step'])):.2f}" cy="{y_for(float(latest['coverage'])):.2f}" r="5" fill="#2563EB" />
  <text x="{left + chart_w / 2:.2f}" y="{height - 35}" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">training step / dataset hook call</text>
  <text x="24" y="{top + chart_h / 2:.2f}" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827" transform="rotate(-90 24 {top + chart_h / 2:.2f})">successful Stage 2 question coverage</text>
  <text x="{left}" y="{height - 15}" font-family="Arial" font-size="12" fill="#6B7280">orange dashed lines mark epoch-end chart writes</text>
</svg>
'''
        latest_path = self.artifact_dir / "coverage_chart.svg"
        latest_path.write_text(svg, encoding="utf-8")

        chart_dir = self.artifact_dir / "coverage_charts"
        chart_dir.mkdir(parents=True, exist_ok=True)
        safe_reason = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in reason).strip("_")
        if not safe_reason:
            safe_reason = "chart"
        snapshot_path = chart_dir / f"coverage_{safe_reason}.svg"
        snapshot_path.write_text(svg, encoding="utf-8")

    def _metrics_snapshot(self, **batch_values: int | float) -> dict[str, int | float]:
        payload = self._summary_payload()
        metrics: dict[str, int | float] = {
            "dynamic_mc/initial_question_count": payload["initial_question_count"],
            "dynamic_mc/stage1_seen_total": payload["stage1_seen_total"],
            "dynamic_mc/stage1_rejected_total": payload["stage1_rejected_total"],
            "dynamic_mc/stage1_duplicate_total": payload["stage1_duplicate_total"],
            "dynamic_mc/stage1_accepted_correct_total": payload["stage1_accepted_correct_total"],
            "dynamic_mc/stage1_accepted_incorrect_total": payload["stage1_accepted_incorrect_total"],
            "dynamic_mc/stage1_accepted_total": payload["stage1_accepted_total"],
            "dynamic_mc/questions_with_correct_candidate": payload["questions_with_correct_candidate"],
            "dynamic_mc/questions_with_incorrect_candidate": payload["questions_with_incorrect_candidate"],
            "dynamic_mc/stage2_queued_total": payload["stage2_queued_total"],
            "dynamic_mc/stage2_inserted_total": payload["stage2_inserted_total"],
            "dynamic_mc/stage2_pending_total": payload["stage2_pending_total"],
            "dynamic_mc/successful_stage2_question_count": payload["successful_stage2_question_count"],
            "dynamic_mc/successful_stage2_question_coverage": payload["successful_stage2_question_coverage"],
        }
        metrics.update({f"dynamic_mc/{key}": value for key, value in batch_values.items()})
        return metrics

    def _decode_responses(self, batch: DataProto) -> list[str]:
        prompt_len = batch.batch["prompts"].shape[-1]
        response_ids = batch.batch["responses"]
        attention_mask = batch.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        responses: list[str] = []
        for index in range(len(batch)):
            valid_len = int(valid_response_lengths[index].item())
            responses.append(self.tokenizer.decode(response_ids[index][:valid_len], skip_special_tokens=True))
        return responses

    def _candidate_from_response(self, *, response: str, extra_info: dict[str, Any]) -> VerifiedCandidate | None:
        requested_role = str(extra_info.get("role_requested", ""))
        gold_answer = str(extra_info.get("gold_answer", ""))
        parsed = extract_generated_answer(response)
        if parsed.normalized_answer is None:
            return None
        is_correct = values_equal(parsed.normalized_answer, gold_answer)
        if requested_role == "candidate":
            role = "correct" if is_correct else "incorrect"
        elif requested_role == "correct":
            if not is_correct:
                return None
            role = "correct"
        elif requested_role == "incorrect":
            if is_correct:
                return None
            role = "incorrect"
        else:
            return None
        return VerifiedCandidate(
            completion=response.strip(),
            final_answer=parsed.normalized_answer,
            role=role,
        )

    def _record_candidate(self, *, question_id: str, extra_info: dict[str, Any], candidate: VerifiedCandidate) -> bool:
        entry = self._candidate_buffer.setdefault(
            question_id,
            {
                "question": str(extra_info.get("question", "")),
                "gold_answer": str(extra_info.get("gold_answer", "")),
                "item_id": str(extra_info.get("item_id", question_id)),
                "correct": [],
                "incorrect": [],
                "seen_correct_completions": set(),
                "seen_incorrect_answers": set(),
                "correct_cursor": 0,
                "incorrect_cursor": 0,
            },
        )
        if candidate.role == "correct":
            if candidate.completion in entry["seen_correct_completions"]:
                return False
            entry["seen_correct_completions"].add(candidate.completion)
            entry["correct"].append(candidate)
            return True
        if candidate.final_answer in entry["seen_incorrect_answers"]:
            return False
        entry["seen_incorrect_answers"].add(candidate.final_answer)
        entry["incorrect"].append(candidate)
        return True

    def _format_stage2_option_completion(self, completion: str) -> str:
        completion = completion.strip()
        if self.stage2_candidate_max_chars <= 0 or len(completion) <= self.stage2_candidate_max_chars:
            return completion

        marker = "FINAL_ANSWER:"
        marker_index = completion.rfind(marker)
        if marker_index == -1:
            return completion[: self.stage2_candidate_max_chars].rstrip() + "\n...[truncated]"

        final_tail = completion[marker_index:].strip()
        head_budget = self.stage2_candidate_max_chars - len(final_tail) - len("\n...\n")
        if head_budget <= 0:
            return final_tail[-self.stage2_candidate_max_chars :]
        return completion[:head_budget].rstrip() + "\n...\n" + final_tail

    def _commit_stage2_candidate_set(self, *, question_id: str, entry: dict[str, Any], incorrect_end: int) -> None:
        self._stage2_counts[question_id] = self._stage2_counts.get(question_id, 0) + 1
        if int(entry.get("correct_cursor", 0)) < len(entry["correct"]):
            entry["correct_cursor"] = int(entry.get("correct_cursor", 0)) + 1
        entry["incorrect_cursor"] = incorrect_end

    def _stage2_record_passes_prompt_filter(self, record: dict[str, Any]) -> bool:
        dataframe = datasets.Dataset.from_list([record])
        return len(self.maybe_filter_out_long_prompts(dataframe)) == 1

    def _build_stage2_record(
        self,
        *,
        question_id: str,
        entry: dict[str, Any],
        commit: bool = True,
    ) -> dict[str, Any] | None:
        if self._stage2_counts.get(question_id, 0) >= self.max_stage2_per_question:
            return None
        if not entry["correct"]:
            return None
        incorrect_start = int(entry.get("incorrect_cursor", 0))
        incorrect_end = incorrect_start + self.incorrect_target_count
        if len(entry["incorrect"]) < incorrect_end:
            return None

        correct_index = int(entry.get("correct_cursor", 0))
        if correct_index >= len(entry["correct"]):
            # Correct solutions are rarer than wrong ones. Reuse the latest verified
            # correct CoT only when new wrong candidates are available.
            correct_index = len(entry["correct"]) - 1
        correct_candidate = entry["correct"][correct_index]
        incorrect_candidates = entry["incorrect"][incorrect_start:incorrect_end]

        choices = [("correct", correct_candidate)] + [("incorrect", value) for value in incorrect_candidates]
        rng = random.Random(f"{self.seed}:{question_id}:{self._stage2_counts.get(question_id, 0)}")
        rng.shuffle(choices)
        options = {
            label: self._format_stage2_option_completion(candidate.completion)
            for label, (_, candidate) in zip(MC_LABELS, choices, strict=True)
        }
        correct_label = next(label for label, (role, _) in zip(MC_LABELS, choices, strict=True) if role == "correct")
        if commit:
            self._commit_stage2_candidate_set(question_id=question_id, entry=entry, incorrect_end=incorrect_end)

        return {
            "data_source": STAGE2_SOURCE,
            "prompt": [{"role": "user", "content": build_mc_onecorrect_prompt(entry["question"], options)}],
            "question_id": question_id,
            "reward_model": {"style": "rule", "ground_truth": correct_label},
            "extra_info": {
                "stage": "stage2_mc",
                "item_id": entry["item_id"],
                "question_id": question_id,
                "question": entry["question"],
                "gold_answer": entry["gold_answer"],
                "role_requested": "",
                "candidate_slot": -1,
                "correct_choice": correct_label,
                "option_roles": {label: role for label, (role, _) in zip(MC_LABELS, choices, strict=True)},
                "option_final_answers": {
                    label: candidate.final_answer for label, (_, candidate) in zip(MC_LABELS, choices, strict=True)
                },
            },
        }

    def _queue_stage2_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        self._pending_stage2_records.extend(records)
        self._stage2_queued_total += len(records)
        for record in records:
            question_id = str(record.get("question_id", ""))
            if question_id:
                self._stage2_question_ids.add(question_id)
            self._append_artifact(
                "stage2_prompts.jsonl",
                {
                    "event": "stage2_prompt_queued",
                    "hook": self._hook_calls,
                    "question_id": question_id,
                    "reward_model": record.get("reward_model", {}),
                    "extra_info": record.get("extra_info", {}),
                    "prompt": record.get("prompt", []),
                },
            )
        self._write_summary()
        print(
            "[GSM8KDynamicMCDataset] "
            f"queued_stage2={len(records)} pending_stage2={len(self._pending_stage2_records)} "
            f"successful_stage2_questions={len(self._stage2_question_ids)}"
        )

    def has_pending_dynamic_rows(self) -> bool:
        return bool(self._pending_stage2_records)

    def _flush_pending_stage2_records(self, *, epoch: int | None = None) -> int:
        if not self._pending_stage2_records:
            return 0
        records = self._pending_stage2_records
        self._pending_stage2_records = []
        dataframe = datasets.Dataset.from_list(records)
        dataframe = self.maybe_filter_out_long_prompts(dataframe)
        if len(dataframe) == 0:
            self._write_summary()
            return 0
        if self.stage2_insert_strategy == "prepend":
            self.dataframe = datasets.concatenate_datasets([dataframe, self.dataframe])
        else:
            self.dataframe = datasets.concatenate_datasets([self.dataframe, dataframe])
        inserted_count = len(dataframe)
        self._inserted_stage2_total += inserted_count
        self._append_artifact(
            "stage2_insertions.jsonl",
            {
                "event": "stage2_rows_inserted",
                "epoch": epoch,
                "inserted_stage2": inserted_count,
                "inserted_stage2_total": self._inserted_stage2_total,
                "dataset_len": len(self.dataframe),
            },
        )
        self._write_summary()
        print(
            "[GSM8KDynamicMCDataset] "
            f"inserted_stage2={inserted_count} strategy={self.stage2_insert_strategy} "
            f"inserted_stage2_total={self._inserted_stage2_total} dataset_len={len(self.dataframe)}"
        )
        return inserted_count

    def on_epoch_end(self, epoch: int) -> int:
        print(
            "[GSM8KDynamicMCDataset] "
            f"epoch_end={epoch} pending_stage2={len(self._pending_stage2_records)}"
        )
        inserted = self._flush_pending_stage2_records(epoch=epoch)
        self._record_coverage_history(event="epoch_end", epoch=epoch)
        if self.coverage_chart_on_epoch:
            self._write_coverage_chart(reason=f"epoch_{epoch:04d}", epoch=epoch)
        return inserted

    def on_batch_end(self, batch: DataProto) -> dict[str, int | float]:
        self._hook_calls += 1
        responses = self._decode_responses(batch)
        new_records: list[dict[str, Any]] = []
        stage1_seen = 0
        accepted_correct = 0
        accepted_incorrect = 0
        rejected = 0
        duplicate = 0
        for index, response in enumerate(responses):
            extra_info = batch[index].non_tensor_batch.get("extra_info", {})
            if not isinstance(extra_info, dict) or extra_info.get("stage") != "stage1_candidate":
                continue
            stage1_seen += 1
            question_id = str(extra_info.get("question_id", ""))
            if not question_id:
                continue
            candidate = self._candidate_from_response(response=response, extra_info=extra_info)
            if candidate is None:
                rejected += 1
                continue
            added = self._record_candidate(question_id=question_id, extra_info=extra_info, candidate=candidate)
            if not added:
                duplicate += 1
                continue
            if candidate.role == "correct":
                accepted_correct += 1
            else:
                accepted_incorrect += 1
            artifact_payload = {
                "event": "stage1_candidate_accepted",
                "hook": self._hook_calls,
                "question_id": question_id,
                "item_id": str(extra_info.get("item_id", "")),
                "candidate_slot": extra_info.get("candidate_slot"),
                "role_requested": str(extra_info.get("role_requested", "")),
                "role": candidate.role,
                "final_answer": candidate.final_answer,
                "gold_answer": str(extra_info.get("gold_answer", "")),
            }
            if self.artifact_include_completions:
                artifact_payload["completion"] = candidate.completion
            self._append_artifact("stage1_candidates.jsonl", artifact_payload)
            if len(new_records) < self.max_new_stage2_per_batch:
                entry = self._candidate_buffer[question_id]
                incorrect_end = int(entry.get("incorrect_cursor", 0)) + self.incorrect_target_count
                stage2 = self._build_stage2_record(question_id=question_id, entry=entry, commit=False)
            else:
                stage2 = None
            if stage2 is not None and self._stage2_record_passes_prompt_filter(stage2):
                self._commit_stage2_candidate_set(question_id=question_id, entry=entry, incorrect_end=incorrect_end)
                new_records.append(stage2)
        self._stage1_seen_total += stage1_seen
        self._stage1_rejected_total += rejected
        self._stage1_duplicate_total += duplicate
        self._accepted_correct_total += accepted_correct
        self._accepted_incorrect_total += accepted_incorrect
        print(
            "[GSM8KDynamicMCDataset] "
            f"hook={self._hook_calls} stage1_seen={stage1_seen} "
            f"accepted_correct={accepted_correct} accepted_incorrect={accepted_incorrect} rejected={rejected} duplicate={duplicate} "
            f"accepted_correct_total={self._accepted_correct_total} "
            f"accepted_incorrect_total={self._accepted_incorrect_total} "
            f"stage2_ready={len(new_records)}"
        )
        self._queue_stage2_records(new_records)
        self._write_summary()
        self._record_coverage_history(event="batch_end")
        if self.coverage_chart_interval > 0 and self._hook_calls % self.coverage_chart_interval == 0:
            self._write_coverage_chart(reason=f"step_{self._hook_calls:06d}")
        return self._metrics_snapshot(
            stage1_seen_batch=stage1_seen,
            stage1_rejected_batch=rejected,
            stage1_duplicate_batch=duplicate,
            stage1_accepted_correct_batch=accepted_correct,
            stage1_accepted_incorrect_batch=accepted_incorrect,
            stage2_ready_batch=len(new_records),
        )


def compute_score(
    data_source: str | None = None,
    solution_str: str | None = None,
    ground_truth: str | None = None,
    extra_info: dict[str, Any] | None = None,
    format_score: float = 0.0,
    score: float = 1.0,
    **kwargs: Any,
) -> dict[str, Any] | list[dict[str, Any]]:
    if data_source is None and "data_sources" in kwargs:
        return compute_score_batched(
            data_sources=kwargs.pop("data_sources"),
            solution_strs=kwargs.pop("solution_strs"),
            ground_truths=kwargs.pop("ground_truths"),
            extra_infos=kwargs.pop("extra_infos"),
            format_score=format_score,
            score=score,
            **kwargs,
        )
    if data_source is None or solution_str is None or ground_truth is None:
        raise ValueError("compute_score requires data_source, solution_str, and ground_truth for single examples.")

    extra_info = extra_info or {}
    stage = extra_info.get("stage")

    if data_source == STAGE2_SOURCE or stage == "stage2_mc":
        result = gsm8k_mc.compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            method=kwargs.get("mc_method", "strict"),
            format_score=format_score,
            score=score,
        )
        return {**result, "stage": "stage2_mc"}

    parsed = extract_generated_answer(solution_str)
    if parsed.normalized_answer is None:
        return {
            "score": 0.0,
            "stage": "stage1_candidate",
            "label": "parsing_error",
            "format_ok": False,
            "pred": "",
            "role_requested": extra_info.get("role_requested"),
        }

    gold_answer = str(extra_info.get("gold_answer") or ground_truth)
    role_requested = str(extra_info.get("role_requested", "correct"))
    is_correct = values_equal(parsed.normalized_answer, gold_answer)
    if role_requested == "incorrect":
        accepted = not is_correct
    else:
        accepted = is_correct
    return {
        "score": score if accepted else format_score,
        "stage": "stage1_candidate",
        "label": "correct" if is_correct else "incorrect",
        "format_ok": True,
        "pred": parsed.normalized_answer,
        "role_requested": role_requested,
        "accepted": accepted,
    }


def compute_score_batched(
    data_sources: list[str] | np.ndarray,
    solution_strs: list[str],
    ground_truths: list[str],
    extra_infos: list[dict[str, Any]],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    return [
        compute_score(
            data_source=str(data_source),
            solution_str=solution_str,
            ground_truth=str(ground_truth),
            extra_info=extra_info,
            **kwargs,
        )
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        )
    ]
