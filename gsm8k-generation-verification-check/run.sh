#!/usr/bin/env bash
set -euo pipefail

python gsm8k-generation-verification-check/run_check.py \
  --model "${MODEL:-Qwen/Qwen2.5-3B-Instruct}" \
  --backend "${BACKEND:-transformers}" \
  --split "${SPLIT:-all}" \
  --num-questions "${NUM_QUESTIONS:-0}" \
  --n-rollouts "${N_ROLLOUTS:-8}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --output-dir "${OUTPUT_DIR:-gsm8k-generation-verification-check/runs/default}" \
  --wandb-project "${WANDB_PROJECT:-multiple_choice_question_study}" \
  --wandb-entity "${WANDB_ENTITY:-}" \
  --wandb-run-name "${WANDB_RUN_NAME:-gsm8k_generation_verification_check}" \
  --wandb-run-id "${WANDB_RUN_ID:-}" \
  --wandb-artifact-name "${WANDB_ARTIFACT_NAME:-}" \
  --wandb-mode "${WANDB_MODE:-online}" \
  --overwrite

