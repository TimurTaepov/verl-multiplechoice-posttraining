from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import asdict, dataclass
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


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_stage1_prompt(question: str) -> str:
    return (
        "Question:\n"
        f"{question.strip()}\n\n"
        "Answer:\n"
        "Let's think step by step."
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
        from vllm import SamplingParams

        formatted = [self._format_prompt(prompt, cfg.use_chat_template) for prompt in prompts]
        sampling = SamplingParams(
            n=1,
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            seed=cfg.seed,
        )
        results = self.llm.generate(formatted, sampling)
        return [result.outputs[0].text.strip() if result.outputs else "" for result in results]


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

    expanded_stage1: list[dict[str, Any]] = []
    stage1_prompts: list[str] = []
    for example in examples:
        for rollout_index in range(args.n_rollouts):
            expanded_stage1.append({**example, "rollout_index": rollout_index})
            stage1_prompts.append(build_stage1_prompt(example["question"]))

    print(f"[diagnostic] loaded_questions={len(examples)} stage1_prompts={len(stage1_prompts)}")
    stage1_outputs = generator.generate(stage1_prompts, stage1_cfg)
    candidates: list[dict[str, Any]] = []
    for candidate_index, (meta, output) in enumerate(zip(expanded_stage1, stage1_outputs, strict=True)):
        label_info = label_candidate(output, meta["gold_answer"], args.parse_method)
        record = {
            "candidate_id": f"{meta['question_id']}:{meta['rollout_index']}",
            "candidate_index": candidate_index,
            **meta,
            "stage1_prompt": stage1_prompts[candidate_index],
            "stage1_response": output,
            **label_info,
        }
        candidates.append(record)
        append_jsonl(stage1_path, record)

    print(f"[diagnostic] stage1_done candidates={len(candidates)}")
    stage2_prompts = [build_stage2_prompt(row["question"], row["stage1_response"]) for row in candidates]
    stage2_outputs = generator.generate(stage2_prompts, stage2_cfg)

    final_records: list[dict[str, Any]] = []
    for row, prompt, output in zip(candidates, stage2_prompts, stage2_outputs, strict=True):
        verifier_verdict = parse_verdict(output)
        verification_correct = verifier_verdict == row["target_verdict"]
        record = {
            **row,
            "stage2_prompt": prompt,
            "stage2_response": output,
            "verifier_verdict": verifier_verdict,
            "verification_correct": verification_correct,
        }
        final_records.append(record)
        append_jsonl(stage2_path, record)

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

