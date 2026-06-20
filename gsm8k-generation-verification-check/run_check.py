from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import datasets

from verl.utils.reward_score import gsm8k as verl_gsm8k


def question_id_from_question(question: str) -> str:
    return hashlib.md5(question.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def iter_progress(iterable: Iterable[Any], *, total: int, desc: str) -> Iterable[Any]:
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True)


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_stage1_prompt(question: str) -> str:
    return (
        "Question:\n"
        f"{question.strip()}\n\n"
        "End your response with exactly one line in this format:\n"
        "FINAL_ANSWER:<number>"
    )


def build_stage2_prompt(question: str, proposed_solution: str) -> str:
    return (
        "Question:\n"
        f"{question.strip()}\n\n"
        "Proposed solution:\n"
        f"{proposed_solution.strip()}\n\n"
        "Is this answer correct?\n"
        "Answer with exactly one word first: correct or incorrect."
    )


def parse_verdict(text: str) -> str:
    head = text.strip().lower()[:500]
    if not head:
        return "unknown"
    if re.search(r"\b(incorrect|wrong|false)\b", head):
        return "incorrect"
    if re.search(r"\bnot\s+(?:mathematically\s+)?correct\b", head):
        return "incorrect"
    if re.match(r"^\s*(no|nope)\b", head):
        return "incorrect"
    if re.search(r"\b(correct|right|true)\b", head):
        return "correct"
    if re.match(r"^\s*(yes|yeah|yep)\b", head):
        return "correct"
    return "unknown"


def normalize_gold_answer(answer_text: str) -> str | None:
    return verl_gsm8k.extract_solution(answer_text, method="strict")


def label_candidate(solution: str, gold_answer: str, parse_method: str) -> dict[str, Any]:
    parsed_answer = verl_gsm8k.extract_solution(solution, method=parse_method)
    if parsed_answer is None:
        return {
            "parsed_answer": None,
            "rule_label": "parse_error",
            "target_verdict": "incorrect",
            "generation_score": 0.0,
        }
    generation_score = float(
        verl_gsm8k.compute_score(
            solution_str=solution,
            ground_truth=gold_answer,
            method=parse_method,
            format_score=0.0,
            score=1.0,
        )
    )
    rule_label = "correct" if generation_score == 1.0 else "incorrect"
    return {
        "parsed_answer": parsed_answer,
        "rule_label": rule_label,
        "target_verdict": "correct" if rule_label == "correct" else "incorrect",
        "generation_score": generation_score,
    }


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    backend: str
    batch_size: int
    max_new_tokens: int
    temperature: float
    top_p: float
    seed: int
    use_chat_template: bool
    torch_dtype: str
    device_map: str
    trust_remote_code: bool


class Generator:
    def generate(self, prompts: Sequence[str], cfg: GenerationConfig) -> list[str]:
        raise NotImplementedError

    def generate_many(self, prompts: Sequence[str], cfg: GenerationConfig, n: int) -> list[list[str]]:
        if n < 1:
            raise ValueError("n must be >= 1")
        grouped = [[] for _ in prompts]
        for rollout_index in range(n):
            rollout_cfg = replace(cfg, seed=cfg.seed + rollout_index)
            outputs = self.generate(prompts, rollout_cfg)
            if len(outputs) != len(prompts):
                raise RuntimeError(f"Generated {len(outputs)} outputs for {len(prompts)} prompts")
            for prompt_index, output in enumerate(outputs):
                grouped[prompt_index].append(output)
        return grouped


class TransformersGenerator(Generator):
    def __init__(self, cfg: GenerationConfig) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model,
            trust_remote_code=cfg.trust_remote_code,
            padding_side="left",
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype: Any = "auto"
        if cfg.torch_dtype != "auto":
            dtype = getattr(torch, cfg.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            trust_remote_code=cfg.trust_remote_code,
            torch_dtype=dtype,
            device_map=cfg.device_map,
        )
        self.model.eval()

    def _format_prompt(self, prompt: str, use_chat_template: bool) -> str:
        if not use_chat_template or not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompts: Sequence[str], cfg: GenerationConfig) -> list[str]:
        formatted = [self._format_prompt(prompt, cfg.use_chat_template) for prompt in prompts]
        outputs: list[str] = []
        do_sample = cfg.temperature > 0.0
        for batch in batched(formatted, cfg.batch_size):
            inputs = self.tokenizer(
                list(batch),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_device = next(self.model.parameters()).device
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            kwargs: dict[str, Any] = {
                "max_new_tokens": cfg.max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if do_sample:
                kwargs["temperature"] = cfg.temperature
                kwargs["top_p"] = cfg.top_p
            with self.torch.inference_mode():
                generated = self.model.generate(**inputs, **kwargs)
            prompt_len = inputs["input_ids"].shape[1]
            texts = self.tokenizer.batch_decode(generated[:, prompt_len:], skip_special_tokens=True)
            outputs.extend(text.strip() for text in texts)
        return outputs


class VLLMGenerator(Generator):
    def __init__(self, cfg: GenerationConfig) -> None:
        from vllm import LLM

        self.llm = LLM(model=cfg.model, trust_remote_code=cfg.trust_remote_code)
        self.tokenizer = self.llm.get_tokenizer()

    def _format_prompt(self, prompt: str, use_chat_template: bool) -> str:
        if not use_chat_template or not hasattr(self.tokenizer, "apply_chat_template"):
            return prompt
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate(self, prompts: Sequence[str], cfg: GenerationConfig) -> list[str]:
        grouped = self.generate_many(prompts, cfg, 1)
        return [outputs[0] if outputs else "" for outputs in grouped]

    def generate_many(self, prompts: Sequence[str], cfg: GenerationConfig, n: int) -> list[list[str]]:
        from vllm import SamplingParams

        if n < 1:
            raise ValueError("n must be >= 1")
        formatted = [self._format_prompt(prompt, cfg.use_chat_template) for prompt in prompts]
        sampling = SamplingParams(
            n=n,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=cfg.seed,
        )
        results = self.llm.generate(formatted, sampling)
        grouped: list[list[str]] = []
        for result in results:
            outputs = [output.text.strip() for output in result.outputs]
            if len(outputs) != n:
                raise RuntimeError(f"vLLM generated {len(outputs)} outputs for one prompt; expected {n}")
            grouped.append(outputs)
        return grouped


def make_generator(cfg: GenerationConfig) -> Generator:
    if cfg.backend == "transformers":
        return TransformersGenerator(cfg)
    if cfg.backend == "vllm":
        return VLLMGenerator(cfg)
    raise ValueError(f"Unknown backend: {cfg.backend}")


def load_gsm8k(split: str, num_questions: int, start_index: int) -> list[dict[str, Any]]:
    split_names = ["train", "test"] if split == "all" else [split]
    examples: list[dict[str, Any]] = []
    global_index = 0
    max_count = None if num_questions <= 0 else num_questions

    for split_name in split_names:
        dataset = datasets.load_dataset("openai/gsm8k", "main", split=split_name)
        for split_index in range(len(dataset)):
            if global_index < start_index:
                global_index += 1
                continue
            if max_count is not None and len(examples) >= max_count:
                return examples
            row = dataset[split_index]
            question = str(row["question"])
            gold_answer = normalize_gold_answer(str(row["answer"]))
            if gold_answer is None:
                global_index += 1
                continue
            examples.append(
                {
                    "dataset_split": split_name,
                    "dataset_index": split_index,
                    "global_index": global_index,
                    "question_id": question_id_from_question(question),
                    "question": question,
                    "gold_answer": gold_answer,
                    "gold_answer_raw": str(row["answer"]),
                }
            )
            global_index += 1
    return examples


def safe_div(num: int | float, den: int | float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    correct = sum(1 for row in records if row["rule_label"] == "correct")
    incorrect = sum(1 for row in records if row["rule_label"] in {"incorrect", "parse_error"})
    parseable = sum(1 for row in records if row["rule_label"] != "parse_error")
    verification_correct = sum(1 for row in records if row["verification_correct"])
    verdict_parseable = sum(1 for row in records if row["verifier_verdict"] != "unknown")
    correct_candidates = [row for row in records if row["rule_label"] == "correct"]
    incorrect_candidates = [row for row in records if row["rule_label"] in {"incorrect", "parse_error"}]
    correct_acc = safe_div(
        sum(1 for row in correct_candidates if row["verifier_verdict"] == "correct"),
        len(correct_candidates),
    )
    incorrect_acc = safe_div(
        sum(1 for row in incorrect_candidates if row["verifier_verdict"] == "incorrect"),
        len(incorrect_candidates),
    )
    balanced = None
    if correct_acc is not None and incorrect_acc is not None:
        balanced = (correct_acc + incorrect_acc) / 2.0
    return {
        "total_candidates": total,
        "generation_correct_count": correct,
        "generation_incorrect_or_parse_error_count": incorrect,
        "parseable_count": parseable,
        "generation_accuracy": safe_div(correct, total),
        "parse_success_rate": safe_div(parseable, total),
        "discrimination_accuracy": safe_div(verification_correct, total),
        "verification_raw_accuracy": safe_div(verification_correct, total),
        "verifier_verdict_parse_rate": safe_div(verdict_parseable, total),
        "accuracy_on_correct_candidates": correct_acc,
        "accuracy_on_incorrect_candidates": incorrect_acc,
        "balanced_verification_accuracy": balanced,
        "verification_minus_generation": (
            None
            if balanced is None or safe_div(correct, total) is None
            else balanced - safe_div(correct, total)
        ),
    }



def upload_wandb_artifact(args: argparse.Namespace, output_dir: Path, metrics: dict[str, Any]) -> None:
    if args.wandb_mode == "disabled":
        return
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("wandb is required because artifact upload is enabled. Install wandb or pass --wandb-mode disabled.") from exc

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name,
        id=args.wandb_run_id or None,
        resume="allow" if args.wandb_run_id else None,
        mode=args.wandb_mode,
        config=vars(args),
    )
    wandb.log({key: value for key, value in metrics.items() if isinstance(value, (int, float)) and value is not None})
    artifact_name = args.wandb_artifact_name or f"{run.name or run.id}_gsm8k_generation_verification_check"
    artifact = wandb.Artifact(
        name=artifact_name,
        type="gsm8k-generation-verification-check",
        metadata={
            "model": args.model,
            "split": args.split,
            "num_questions": metrics.get("num_questions"),
            "n_rollouts": args.n_rollouts,
            "backend": args.backend,
        },
    )
    artifact.add_dir(str(output_dir))
    run.log_artifact(artifact)
    run.finish()


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = output_dir / "stage1_candidates.jsonl"
    stage2_path = output_dir / "stage2_discrimination.jsonl"
    for path in (stage1_path, stage2_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists. Pass --overwrite to replace it.")
        if path.exists():
            path.unlink()

    examples = load_gsm8k(args.split, args.num_questions, args.start_index)
    if not examples:
        raise RuntimeError("No GSM8K examples loaded.")

    stage1_cfg = GenerationConfig(
        model=args.model,
        backend=args.backend,
        batch_size=args.batch_size,
        max_new_tokens=args.stage1_max_new_tokens,
        temperature=args.stage1_temperature,
        top_p=args.stage1_top_p,
        seed=args.seed,
        use_chat_template=not args.no_chat_template,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    stage2_cfg = GenerationConfig(
        model=args.model,
        backend=args.backend,
        batch_size=args.batch_size,
        max_new_tokens=args.stage2_max_new_tokens,
        temperature=args.stage2_temperature,
        top_p=args.stage2_top_p,
        seed=args.seed,
        use_chat_template=not args.no_chat_template,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )
    config_payload = {
        **vars(args),
        "reward_parser": "verl.utils.reward_score.gsm8k",
        "stage1_prompt_template": build_stage1_prompt("<question>"),
        "stage2_prompt_template": build_stage2_prompt("<question>", "<proposed_solution>"),
    }
    write_json(output_dir / "config.json", config_payload)

    generator = make_generator(stage1_cfg)
    progress_path = output_dir / "progress.json"
    started_at = time.time()
    monotonic_start = time.monotonic()

    stage1_prompts = [build_stage1_prompt(example["question"]) for example in examples]
    stage1_total = len(examples) * args.n_rollouts
    stage1_batches = (len(stage1_prompts) + stage1_cfg.batch_size - 1) // stage1_cfg.batch_size
    label_counts = {"correct": 0, "incorrect": 0, "parse_error": 0}
    print(
        f"[diagnostic] loaded_questions={len(examples)} "
        f"stage1_prompts={len(stage1_prompts)} stage1_candidates={stage1_total}",
        flush=True,
    )
    candidates: list[dict[str, Any]] = []
    write_progress(
        progress_path,
        {
            "stage": "stage1",
            "started_at": started_at,
            "elapsed_sec": 0.0,
            "stage1_completed": 0,
            "stage1_total": stage1_total,
            "stage2_completed": 0,
            "stage2_total": None,
            "output_dir": str(output_dir),
            "stage1_path": str(stage1_path),
            "stage2_path": str(stage2_path),
        },
    )

    for batch_start in iter_progress(range(0, len(stage1_prompts), stage1_cfg.batch_size), total=stage1_batches, desc="stage1"):
        batch_end = min(batch_start + stage1_cfg.batch_size, len(stage1_prompts))
        batch_prompts = stage1_prompts[batch_start:batch_end]
        batch_examples = examples[batch_start:batch_end]
        batch_outputs = generator.generate_many(batch_prompts, stage1_cfg, args.n_rollouts)
        if len(batch_outputs) != len(batch_examples):
            raise RuntimeError(f"Stage 1 generated output groups for {len(batch_outputs)} prompts; expected {len(batch_examples)}")
        for example, prompt, rollout_outputs in zip(batch_examples, batch_prompts, batch_outputs, strict=True):
            if len(rollout_outputs) != args.n_rollouts:
                raise RuntimeError(f"Stage 1 generated {len(rollout_outputs)} rollouts; expected {args.n_rollouts}")
            for rollout_index, output in enumerate(rollout_outputs):
                candidate_index = len(candidates)
                label_info = label_candidate(output, example["gold_answer"], args.parse_method)
                label_counts[label_info["rule_label"]] += 1
                record = {
                    "candidate_id": f"{example['question_id']}:{rollout_index}",
                    "candidate_index": candidate_index,
                    **example,
                    "rollout_index": rollout_index,
                    "stage1_prompt": prompt,
                    "stage1_response": output,
                    **label_info,
                }
                candidates.append(record)
                append_jsonl(stage1_path, record)
        elapsed = time.monotonic() - monotonic_start
        write_progress(
            progress_path,
            {
                "stage": "stage1",
                "started_at": started_at,
                "elapsed_sec": elapsed,
                "stage1_completed": len(candidates),
                "stage1_total": stage1_total,
                "stage1_label_counts": label_counts,
                "stage2_completed": 0,
                "stage2_total": None,
                "output_dir": str(output_dir),
                "stage1_path": str(stage1_path),
                "stage2_path": str(stage2_path),
            },
        )
        print(
            f"[progress] stage1 {len(candidates)}/{stage1_total} "
            f"correct={label_counts['correct']} incorrect={label_counts['incorrect']} "
            f"parse_error={label_counts['parse_error']} elapsed_sec={elapsed:.1f}",
            flush=True,
        )

    print(f"[diagnostic] stage1_done candidates={len(candidates)}", flush=True)
    stage2_total = len(candidates)
    stage2_batches = (stage2_total + stage2_cfg.batch_size - 1) // stage2_cfg.batch_size
    final_records: list[dict[str, Any]] = []
    verdict_counts = {"correct": 0, "incorrect": 0, "unknown": 0}
    verification_correct_count = 0
    for batch_start in iter_progress(range(0, stage2_total, stage2_cfg.batch_size), total=stage2_batches, desc="stage2"):
        batch_end = min(batch_start + stage2_cfg.batch_size, stage2_total)
        batch_rows = candidates[batch_start:batch_end]
        batch_prompts = [build_stage2_prompt(row["question"], row["stage1_response"]) for row in batch_rows]
        batch_outputs = generator.generate(batch_prompts, stage2_cfg)
        if len(batch_outputs) != len(batch_rows):
            raise RuntimeError(f"Stage 2 generated {len(batch_outputs)} outputs for {len(batch_rows)} prompts")
        for row, prompt, output in zip(batch_rows, batch_prompts, batch_outputs, strict=True):
            verifier_verdict = parse_verdict(output)
            verdict_counts[verifier_verdict] += 1
            verification_correct = verifier_verdict == row["target_verdict"]
            verification_correct_count += int(verification_correct)
            record = {
                **row,
                "stage2_prompt": prompt,
                "stage2_response": output,
                "verifier_verdict": verifier_verdict,
                "verification_correct": verification_correct,
            }
            final_records.append(record)
            append_jsonl(stage2_path, record)
        elapsed = time.monotonic() - monotonic_start
        write_progress(
            progress_path,
            {
                "stage": "stage2",
                "started_at": started_at,
                "elapsed_sec": elapsed,
                "stage1_completed": stage1_total,
                "stage1_total": stage1_total,
                "stage1_label_counts": label_counts,
                "stage2_completed": len(final_records),
                "stage2_total": stage2_total,
                "stage2_verdict_counts": verdict_counts,
                "stage2_verification_correct_count": verification_correct_count,
                "output_dir": str(output_dir),
                "stage1_path": str(stage1_path),
                "stage2_path": str(stage2_path),
            },
        )
        print(
            f"[progress] stage2 {len(final_records)}/{stage2_total} "
            f"verdict_correct={verdict_counts['correct']} verdict_incorrect={verdict_counts['incorrect']} "
            f"unknown={verdict_counts['unknown']} verification_correct={verification_correct_count} "
            f"elapsed_sec={elapsed:.1f}",
            flush=True,
        )

    metrics = compute_metrics(final_records)
    metrics["num_questions"] = len(examples)
    metrics["n_rollouts"] = args.n_rollouts
    metrics["split"] = args.split
    metrics["model"] = args.model
    metrics["backend"] = args.backend
    write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    upload_wandb_artifact(args, output_dir, metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GSM8K generation-vs-discrimination diagnostic. No training.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--split", default="all", help="GSM8K split: train, test, or all. Default: all")
    parser.add_argument("--num-questions", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--n-rollouts", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--parse-method", choices=["strict", "flexible"], default="flexible")
    parser.add_argument("--stage1-max-new-tokens", type=int, default=512)
    parser.add_argument("--stage1-temperature", type=float, default=1.0)
    parser.add_argument("--stage1-top-p", type=float, default=0.95)
    parser.add_argument("--stage2-max-new-tokens", type=int, default=256)
    parser.add_argument("--stage2-temperature", type=float, default=0.0)
    parser.add_argument("--stage2-top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="gsm8k-generation-verification-check/runs/default")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--wandb-project", default="multiple_choice_question_study")
    parser.add_argument("--wandb-entity", default="")
    parser.add_argument("--wandb-run-name", default="gsm8k_generation_verification_check")
    parser.add_argument("--wandb-run-id", default="")
    parser.add_argument("--wandb-artifact-name", default="")
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    args = parser.parse_args()
    if args.num_questions < 0:
        parser.error("--num-questions must be >= 0; use 0 for all selected questions")
    if args.start_index < 0:
        parser.error("--start-index must be >= 0")
    if args.n_rollouts < 1:
        parser.error("--n-rollouts must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        sys.exit(130)
