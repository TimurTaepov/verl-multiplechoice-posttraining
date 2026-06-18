# GSM8K Generation Verification Check

No training happens here.

This pipeline checks whether the same base model is better at:

1. generating a correct GSM8K solution from the original question;
2. discriminating whether a generated CoT solution is correct or incorrect.

The GSM8K answer parsing/scoring calls `verl.utils.reward_score.gsm8k` directly, so it uses the same parser/reward path as training.

Run:

```bash
python gsm8k-generation-verification-check/run_check.py --model Qwen/Qwen2.5-3B-Instruct --split all --num-questions 0 --n-rollouts 8 --backend transformers --output-dir gsm8k-generation-verification-check/runs/qwen25_3b_all_n8
```

Main outputs:

```text
config.json
stage1_candidates.jsonl
stage2_discrimination.jsonl
metrics.json
```

