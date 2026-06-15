# Dynamic MC Two-Stage Pipeline Report

This report traces the current VERL-native two-stage GSM8K multiple-choice training pipeline from the launcher script down to the custom dataset, reward function, VERL trainer hooks, storage locations, and post-run artifact upload.

The point of the report is file ownership and call flow. Calling VERL files is expected and fine: VERL is the training engine. The custom research logic should be understandable from the `gsm8k-builder` side, while the VERL files listed here are the places where the pipeline enters or extends VERL.

Scope:

- Included: custom project files and VERL files that are directly involved in the current two-stage pipeline.
- Omitted: generic third-party library internals such as `datasets`, `torch`, `wandb`, `ray`, `vllm`, and Hugging Face internals unless a custom/VERL file calls into them in a way that matters for the pipeline.
- Current entry point: `bash/20260604_gsm8k_dynamic_mc_verl_native.sh`.

## 1. High-Level Tree

```text
bash/20260604_gsm8k_dynamic_mc_verl_native.sh
|
|-- creates paths and runtime env
|   |-- PYTHONPATH includes gsm8k-builder/src and repo root
|   |-- seed output: gsm8k-builder/runs_verl_native_dynamic_mc/seed/*.parquet
|   |-- log output: logs/<run_id>.log
|   |-- checkpoint output: checkpoints/dynamic_mc/<run_id>/
|   |-- artifact output: gsm8k-builder/runs_verl_native_dynamic_mc/artifacts/<run_id>/
|
|-- calls gsm8k-builder/create_dynamic_mc_seed.py for train seed parquet
|   |-- imports reliable_gsm8k.verl_dynamic_mc.make_stage1_records_from_gsm8k_example
|   |-- imports reliable_gsm8k.parsing.parse_gold_answer indirectly
|   |-- writes train Stage 1 parquet
|
|-- calls gsm8k-builder/create_dynamic_mc_seed.py for val seed parquet
|   |-- same code path, with val partition
|   |-- writes validation Stage 1 parquet
|
|-- calls python -m verl.trainer.main_ppo
|   |
|   |-- verl/trainer/main_ppo.py
|   |   |-- hydra loads VERL PPO config
|   |   |-- TaskRunner.run builds tokenizer, reward manager, datasets, trainer
|   |   |-- create_rl_dataset loads custom dataset class
|   |
|   |-- verl/trainer/ppo/reward.py
|   |   |-- load_reward_manager loads custom reward function
|   |   |-- custom_reward_function.path points to gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py
|   |
|   |-- gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py
|   |   |-- GSM8KDynamicMCDataset extends VERL RLHFDataset
|   |   |-- compute_score handles Stage 1 and Stage 2 rewards
|   |   |-- Stage 1 rollouts are parsed and buffered
|   |   |-- Stage 2 MC prompts are built and queued
|   |   |-- local dynamic artifacts are written
|   |
|   |-- verl/trainer/ppo/ray_trainer.py
|       |-- RayPPOTrainer train loop runs rollout/reward/update
|       |-- calls train_dataset.on_batch_end(batch)
|       |-- logs dynamic_mc/* metrics
|       |-- calls train_dataset.on_epoch_end(epoch)
|       |-- rebuilds train dataloader after Stage 2 rows are inserted
|
|-- after VERL exits, optionally calls gsm8k-builder/upload_dynamic_mc_artifacts.py
    |-- uploads local artifact folder to W&B if enabled
```

## 2. Entry Point

### File

```text
bash/20260604_gsm8k_dynamic_mc_verl_native.sh
```

### Main responsibilities

This is the current main launcher. It does not itself train the model. It prepares the data files and then starts VERL once.

It sets:

```text
REPO_ROOT=<repo root>
BUILDER_DIR=<repo root>/gsm8k-builder
PYTHONPATH=$BUILDER_DIR/src:$REPO_ROOT:$PYTHONPATH
```

This matters because the custom package still lives under:

```text
gsm8k-builder/src/reliable_gsm8k/
```

The launcher then defines defaults for:

```text
RUN_ID
MODEL_PATH
SPLIT
TRAIN_PARTITION
NUM_SAMPLES
VAL_SOURCE_SPLIT
VAL_PARTITION
VAL_NUM_SAMPLES
VAL_HOLDOUT_SIZE
DATA_SPLIT_SEED
STAGE1_PROMPT_COUNT
ROLLOUT_N
TRAIN_BATCH_SIZE
GEN_BATCH_SIZE
PPO_MINI_BATCH_SIZE
TOTAL_EPOCHS
ARTIFACT_DIR
CHECKPOINT_DIR
LOG_FILE
TRAINER_LOGGER
WANDB_ARTIFACTS
```

### Important output paths created by the launcher

```text
SEED_DIR=$BUILDER_DIR/runs_verl_native_dynamic_mc/seed
SEED_FILE=$SEED_DIR/${RUN_ID}_${SPLIT}_${TRAIN_PARTITION}_stage1.parquet
VAL_FILE=$SEED_DIR/${RUN_ID}_${VAL_PARTITION}_stage1.parquet
CHECKPOINT_DIR=$REPO_ROOT/checkpoints/dynamic_mc/$RUN_ID
LOG_FILE=$REPO_ROOT/logs/$RUN_ID.log
ARTIFACT_DIR=$BUILDER_DIR/runs_verl_native_dynamic_mc/artifacts/$RUN_ID
```

### Commands built by the launcher

The launcher builds three important commands:

```text
CREATE_SEED_CMD
CREATE_VAL_CMD
VERL_CMD
```

`CREATE_SEED_CMD` makes the training Stage 1 seed parquet.

`CREATE_VAL_CMD` makes the validation Stage 1 seed parquet unless `VAL_FILE` is provided by the caller.

`VERL_CMD` starts the actual VERL run:

```text
python -m verl.trainer.main_ppo ...
```

The launcher also optionally calls the W&B artifact uploader after VERL exits.

## 3. Seed Parquet Creation

### File

```text
gsm8k-builder/create_dynamic_mc_seed.py
```

### Called from

```text
bash/20260604_gsm8k_dynamic_mc_verl_native.sh
```

### Purpose

This script converts GSM8K questions into Stage 1 VERL training rows. It writes Parquet files consumed by VERL.

It is called once for training and once for validation.

### Dataset partition logic

The script supports logical partitions:

```text
all
train
val
test
```

For the current launcher defaults:

```text
training:   --split train --partition train
validation: --split train --partition val
```

The train/val split is deterministic:

```text
validation_size = VAL_HOLDOUT_SIZE, default 512
split_seed = DATA_SPLIT_SEED, default 7
```

The script preserves the original GSM8K source index by adding `_source_index`. This is used so `item_id` stays tied to the original GSM8K index after partitioning.

### Custom function called

```text
reliable_gsm8k.verl_dynamic_mc.make_stage1_records_from_gsm8k_example
```

This lives in:

```text
gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py
```

### Output schema produced

Each record has the fields needed by VERL and the custom dynamic dataset:

```text
data_source
prompt
question_id
reward_model
extra_info
```

For Stage 1 rows:

```text
data_source = gsm8k_dynamic_mc_stage1
reward_model.ground_truth = normalized GSM8K gold answer
extra_info.stage = stage1_candidate
extra_info.question_id = md5(question)
extra_info.question = original question
extra_info.gold_answer = normalized gold answer
extra_info.role_requested = candidate | correct | incorrect
extra_info.candidate_slot = stage1 prompt slot
```

The default path uses neutral candidate prompts:

```text
STAGE1_PROMPT_MODE=neutral
```

That means Stage 1 prompts ask the model to solve the question. They do not ask the model to intentionally produce wrong answers by default.

## 4. Custom Dynamic Dataset and Reward File

### File

```text
gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py
```

This is the central custom file for the current pipeline.

It contains:

```text
question_id_from_question
build_stage1_candidate_prompt
make_stage1_record
make_stage1_records_for_question
make_stage1_records_from_gsm8k_example
VerifiedCandidate
GSM8KDynamicMCDataset
compute_score
compute_score_batched
```

## 5. Stage 1 Record Construction

### File

```text
gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py
```

### Key functions

```text
question_id_from_question(question)
make_stage1_record(...)
make_stage1_records_for_question(...)
make_stage1_records_from_gsm8k_example(...)
```

### Parser dependency

```text
gsm8k-builder/src/reliable_gsm8k/parsing.py
```

Used function:

```text
parse_gold_answer(answer_text)
```

`parse_gold_answer` calls the shared VERL GSM8K parser:

```text
verl/utils/reward_score/gsm8k.py
```

Specifically, it uses strict GSM8K extraction from gold answers containing:

```text
#### <answer>
```

### Prompt format

Stage 1 neutral prompt asks for:

```text
REASONING: <step-by-step solution>
FINAL_ANSWER: <final numeric answer>
```

This prompt is built directly in:

```text
build_stage1_candidate_prompt(question)
```

Inside:

```text
gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py
```

There are also role-specific prompt builders:

```text
build_stage1_correct_prompt
build_stage1_incorrect_prompt
```

These are only used if `STAGE1_PROMPT_MODE=role`.

## 6. VERL Launch

### Command built in launcher

```text
python -m verl.trainer.main_ppo
```

### Important overrides passed by the launcher

```text
algorithm.adv_estimator=grpo

data.train_files=$SEED_FILE
data.val_files=$VAL_FILE
data.prompt_key=prompt
data.reward_fn_key=data_source

data.custom_cls.path=pkg://reliable_gsm8k.verl_dynamic_mc
data.custom_cls.name=GSM8KDynamicMCDataset

custom_reward_function.path=$BUILDER_DIR/src/reliable_gsm8k/verl_dynamic_mc.py
custom_reward_function.name=compute_score

val_custom_reward_function.path=$BUILDER_DIR/src/reliable_gsm8k/verl_dynamic_mc.py
val_custom_reward_function.name=compute_score
```

This is the key VERL-native bridge:

- VERL loads our custom dataset class.
- VERL loads our custom reward function.
- The model stays inside the VERL run.
- Dynamic Stage 2 rows are injected by dataset hooks, not by restarting the job.

## 7. VERL Entry and Dataset Loading

### File

```text
verl/trainer/main_ppo.py
```

### Important functions/classes

```text
main(config)
run_ppo(config)
TaskRunner.run(config)
create_rl_dataset(...)
```

### Flow

`python -m verl.trainer.main_ppo` enters:

```text
verl/trainer/main_ppo.py
```

Hydra loads the VERL PPO config:

```text
verl/trainer/config/ppo_trainer.yaml
```

Then `TaskRunner.run(config)` does the setup:

```text
load_reward_manager(...)
create_rl_dataset(... train ...)
create_rl_dataset(... val ...)
RayPPOTrainer(...)
trainer.fit()
```

### Custom dataset loading

In `create_rl_dataset`, VERL checks:

```text
data.custom_cls.path
data.custom_cls.name
```

The launcher sets:

```text
data.custom_cls.path=pkg://reliable_gsm8k.verl_dynamic_mc
data.custom_cls.name=GSM8KDynamicMCDataset
```

So VERL imports:

```text
gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py
```

and instantiates:

```text
GSM8KDynamicMCDataset
```

`GSM8KDynamicMCDataset` subclasses:

```text
verl/utils/dataset/rl_dataset.py::RLHFDataset
```

## 8. VERL Reward Loading

### File

```text
verl/trainer/ppo/reward.py
```

### Important function

```text
load_reward_manager(config, tokenizer, ...)
```

### Custom reward function

The launcher sets:

```text
custom_reward_function.path=$BUILDER_DIR/src/reliable_gsm8k/verl_dynamic_mc.py
custom_reward_function.name=compute_score
```

So VERL loads:

```text
gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py::compute_score
```

The same custom reward function is used for validation through:

```text
val_custom_reward_function.path
val_custom_reward_function.name
```

## 9. Stage 1 Runtime: Rollout, Reward, Buffering

### VERL trainer file

```text
verl/trainer/ppo/ray_trainer.py
```

This file runs the actual PPO/GRPO loop.

During a train batch, VERL:

```text
1. samples prompts from train_dataset
2. asks actor/rollout worker to generate responses
3. computes reward through custom compute_score
4. computes GRPO advantages
5. updates the actor
6. collects metrics
7. calls train_dataset.on_batch_end(batch)
```

### Custom reward: Stage 1

File:

```text
gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py
```

Function:

```text
compute_score(...)
```

For Stage 1 rows:

```text
data_source = gsm8k_dynamic_mc_stage1
extra_info.stage = stage1_candidate
```

`compute_score` calls:

```text
extract_generated_answer(solution_str)
values_equal(parsed_answer, gold_answer)
```

These live in:

```text
gsm8k-builder/src/reliable_gsm8k/parsing.py
```

The parser delegates numeric normalization/equality to:

```text
verl/utils/reward_score/gsm8k.py
```

Stage 1 reward behavior:

```text
parse failure -> score 0
correct numeric answer -> score 1
wrong numeric answer -> score 0
```

If `role_requested=incorrect`, the accepted reward direction changes, but the default current launcher uses:

```text
STAGE1_PROMPT_MODE=neutral
role_requested=candidate
```

### Custom dataset hook: Stage 1 buffering

File:

```text
gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py
```

Class:

```text
GSM8KDynamicMCDataset
```

Method called by VERL:

```text
on_batch_end(batch)
```

Called from:

```text
verl/trainer/ppo/ray_trainer.py
```

Inside `on_batch_end`:

```text
_decode_responses(batch)
_candidate_from_response(response, extra_info)
_record_candidate(question_id, extra_info, candidate)
_build_stage2_record(...) if enough candidates exist
_queue_stage2_records(...)
_write_summary()
return dynamic_mc/* metrics
```

Candidate buffering rules:

```text
correct bucket: stores unique correct completions
incorrect bucket: stores unique wrong final answers
```

Stage 2 eligibility rule:

```text
at least 1 correct candidate
at least 3 distinct incorrect candidates
```

The current implementation tracks duplicate candidates separately:

```text
dynamic_mc/stage1_duplicate_total
dynamic_mc/stage1_duplicate_batch
```

## 10. Stage 2 Prompt Construction

### File

```text
gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py
```

### Function

```text
_build_stage2_record(...)
```

### Prompt builder dependency

```text
gsm8k-builder/src/reliable_gsm8k/prompts.py
```

Used function:

```text
build_mc_onecorrect_prompt(question, options)
```

Stage 2 prompt format:

```text
Question:
<original GSM8K question>

Options:
A. <candidate CoT from Stage 1>
B. <candidate CoT from Stage 1>
C. <candidate CoT from Stage 1>
D. <candidate CoT from Stage 1>

Return exactly one line: #### <letter>
Choose the letter of the only correct numeric answer.
```

Stage 2 row fields:

```text
data_source = gsm8k_dynamic_mc_stage2
prompt = MC prompt
question_id = same md5(question)
reward_model.ground_truth = correct option letter
extra_info.stage = stage2_mc
extra_info.option_roles = A/B/C/D -> correct or incorrect
extra_info.option_final_answers = final numeric answer per option
```

Stage 2 candidate order is randomized deterministically using:

```text
random.Random(f"{seed}:{question_id}:{stage2_count}")
```

## 11. Stage 2 Insertion into VERL Dataset

### VERL trainer file

```text
verl/trainer/ppo/ray_trainer.py
```

### Custom methods called

At batch end:

```text
train_dataset.on_batch_end(batch)
```

At epoch end:

```text
train_dataset.on_epoch_end(epoch)
```

In `GSM8KDynamicMCDataset.on_epoch_end`, pending Stage 2 rows are flushed into the train dataset:

```text
_flush_pending_stage2_records(epoch=epoch)
```

Then VERL rebuilds the train dataloader:

```text
_rebuild_train_dataloader_for_dynamic_dataset(inserted_rows)
```

This is what makes Stage 2 rows trainable without restarting VERL.

The launcher sets:

```text
+data.dynamic_mc.stage2_insert_strategy=prepend
```

So Stage 2 rows are inserted before existing rows when flushed.

## 12. Stage 2 Runtime Reward

### Custom reward function

```text
gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py::compute_score
```

For Stage 2 rows:

```text
data_source = gsm8k_dynamic_mc_stage2
extra_info.stage = stage2_mc
```

`compute_score` delegates to VERL MC scorer:

```text
verl/utils/reward_score/gsm8k_mc.py::compute_score
```

The MC scorer expects the model answer to contain:

```text
#### A
#### B
#### C
#### D
```

Current method passed by `verl_dynamic_mc.py` is:

```text
method="strict"
```

Stage 2 reward behavior:

```text
missing strict choice -> score 0
wrong choice -> score 0
correct choice -> score 1
```

## 13. Dynamic Metrics

### Produced by

```text
gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py::GSM8KDynamicMCDataset._metrics_snapshot
```

### Added to VERL metrics in

```text
verl/trainer/ppo/ray_trainer.py
```

The trainer calls:

```text
dynamic_metrics = self.train_dataset.on_batch_end(batch)
metrics.update(dynamic_metrics)
logger.log(data=metrics, step=self.global_steps)
```

### Important metric keys

```text
dynamic_mc/initial_question_count
dynamic_mc/stage1_seen_total
dynamic_mc/stage1_rejected_total
dynamic_mc/stage1_duplicate_total
dynamic_mc/stage1_accepted_correct_total
dynamic_mc/stage1_accepted_incorrect_total
dynamic_mc/stage1_accepted_total
dynamic_mc/questions_with_correct_candidate
dynamic_mc/questions_with_incorrect_candidate
dynamic_mc/stage2_queued_total
dynamic_mc/stage2_inserted_total
dynamic_mc/stage2_pending_total
dynamic_mc/successful_stage2_question_count
dynamic_mc/successful_stage2_question_coverage
```

Batch-local metric keys include:

```text
dynamic_mc/stage1_seen_batch
dynamic_mc/stage1_rejected_batch
dynamic_mc/stage1_duplicate_batch
dynamic_mc/stage1_accepted_correct_batch
dynamic_mc/stage1_accepted_incorrect_batch
dynamic_mc/stage2_ready_batch
```

## 14. Local Artifacts Written During Training

### Directory

```text
gsm8k-builder/runs_verl_native_dynamic_mc/artifacts/<run_id>/
```

Configured by launcher as:

```text
ARTIFACT_DIR
+data.dynamic_mc.artifact_dir=$ARTIFACT_DIR
```

### Written by

```text
gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py
```

### Files

```text
stage1_candidates.jsonl
stage2_prompts.jsonl
stage2_insertions.jsonl
dynamic_mc_summary.json
```

### `stage1_candidates.jsonl`

Written in:

```text
GSM8KDynamicMCDataset.on_batch_end
```

Contains unique accepted Stage 1 candidates only.

Typical fields:

```text
event
hook
question_id
item_id
candidate_slot
role_requested
role
final_answer
gold_answer
completion  # included when ARTIFACT_INCLUDE_COMPLETIONS=1
```

### `stage2_prompts.jsonl`

Written in:

```text
GSM8KDynamicMCDataset._queue_stage2_records
```

Contains Stage 2 MC rows when they are queued.

Typical fields:

```text
event
hook
question_id
reward_model
extra_info
prompt
```

### `stage2_insertions.jsonl`

Written in:

```text
GSM8KDynamicMCDataset._flush_pending_stage2_records
```

Contains insertion events at epoch end.

Typical fields:

```text
event
epoch
inserted_stage2
inserted_stage2_total
dataset_len
```

### `dynamic_mc_summary.json`

Written repeatedly by:

```text
GSM8KDynamicMCDataset._write_summary
```

Contains the latest counters and coverage values.

Key fields:

```text
initial_question_count
hook_calls
stage1_seen_total
stage1_rejected_total
stage1_duplicate_total
stage1_accepted_correct_total
stage1_accepted_incorrect_total
stage1_accepted_total
questions_with_correct_candidate
questions_with_incorrect_candidate
candidate_buffer_question_count
stage2_queued_total
stage2_inserted_total
stage2_pending_total
successful_stage2_question_count
successful_stage2_question_coverage
```

## 15. Logs and Checkpoints

### Logs

Configured in launcher:

```text
LOG_FILE=$REPO_ROOT/logs/$RUN_ID.log
```

When `DRY_RUN != 1`, the launcher redirects stdout/stderr through `tee` into this file.

### Checkpoints

Configured in launcher:

```text
CHECKPOINT_DIR=$REPO_ROOT/checkpoints/dynamic_mc/$RUN_ID
trainer.default_local_dir=$CHECKPOINT_DIR
```

The launcher asks VERL to save:

```text
actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra","hf_model"]
```

Checkpoints are not the on-policy mechanism. They are for recovery/inspection. The actor stays inside the running VERL process during training.

## 16. W&B Artifact Upload

### File

```text
gsm8k-builder/upload_dynamic_mc_artifacts.py
```

### Called from

```text
bash/20260604_gsm8k_dynamic_mc_verl_native.sh
```

### Condition

The launcher uploads artifacts if:

```text
WANDB_ARTIFACTS=1
```

or if:

```text
WANDB_ARTIFACTS=auto
TRAINER_LOGGER contains wandb
```

### Uploaded artifact contents

Always uploads:

```text
ARTIFACT_DIR as dynamic_mc_artifacts
```

Also attaches existing input/log files:

```text
seed parquet
validation seed parquet
run log
```

Checkpoints are uploaded only if:

```text
WANDB_UPLOAD_CHECKPOINTS=1
```

Default:

```text
WANDB_UPLOAD_CHECKPOINTS=0
```

## 17. Smoke Log Checker

### File

```text
gsm8k-builder/check_dynamic_mc_smoke_log.py
```

### Purpose

This is not part of training. It is an after-run sanity checker for logs.

It checks the log for evidence that:

```text
dataset hook ran
Stage 1 rollouts were seen
correct candidates were accepted
incorrect candidates were accepted
Stage 2 rows were queued
epoch-end promotion ran
Stage 2 rows were inserted
train dataloader was rebuilt
```

Example:

```text
python gsm8k-builder/check_dynamic_mc_smoke_log.py logs/<run_id>.log
```

## 18. Files Involved in Current Pipeline

### Current launcher

```text
bash/20260604_gsm8k_dynamic_mc_verl_native.sh
```

### Current custom pipeline files

```text
gsm8k-builder/create_dynamic_mc_seed.py
gsm8k-builder/upload_dynamic_mc_artifacts.py
gsm8k-builder/check_dynamic_mc_smoke_log.py
gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py
gsm8k-builder/src/reliable_gsm8k/parsing.py
gsm8k-builder/src/reliable_gsm8k/prompts.py
```

### Shared VERL reward files used by custom code

```text
verl/utils/reward_score/gsm8k.py
verl/utils/reward_score/gsm8k_mc.py
```

### VERL trainer/config files directly touched by the pipeline

```text
verl/trainer/main_ppo.py
verl/trainer/ppo/reward.py
verl/trainer/ppo/ray_trainer.py
verl/utils/dataset/rl_dataset.py
verl/trainer/config/ppo_trainer.yaml
verl/trainer/config/actor/actor.yaml
verl/trainer/config/rollout/rollout.yaml
verl/trainer/config/algorithm.py
```

The main custom modification inside VERL is in:

```text
verl/trainer/ppo/ray_trainer.py
```

That file now calls dynamic dataset hooks:

```text
train_dataset.on_batch_end(batch)
train_dataset.on_epoch_end(epoch)
```

and rebuilds the train dataloader when Stage 2 rows are inserted.

## 19. Files Nearby but Not Part of This Current Launcher

These files may be useful historically or for other experiments, but they are not the current two-stage VERL-native entry path unless called separately:

```text
bash/20260327_gsm8k_mc_sampled.sh
bash/20260529_gsm8k_mc_sampled.sh
gsm8k-builder/run_build.py
gsm8k-builder/run_on_policy_loop.py
gsm8k-builder/src/reliable_gsm8k/pipeline.py
gsm8k-builder/src/reliable_gsm8k/backends.py
gsm8k-builder/src/reliable_gsm8k/profiles.py
evals/oe_mc_eval_05_02_26/*.py
examples/data_preprocess/gsm8k_mc_sampled.py
```

Important note: `gsm8k-builder/src/reliable_gsm8k/backends.py` is used by older/offline generation paths, not by the current `20260604_gsm8k_dynamic_mc_verl_native.sh` VERL-native run.

## 20. Current Data Flow in One Pass

```text
1. User runs bash/20260604_gsm8k_dynamic_mc_verl_native.sh

2. Launcher creates train seed parquet:
   gsm8k-builder/create_dynamic_mc_seed.py
   -> gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py
   -> gsm8k-builder/src/reliable_gsm8k/parsing.py
   -> verl/utils/reward_score/gsm8k.py
   -> writes SEED_FILE

3. Launcher creates val seed parquet:
   same path as above
   -> writes VAL_FILE

4. Launcher starts VERL:
   python -m verl.trainer.main_ppo

5. VERL main loads custom reward:
   verl/trainer/ppo/reward.py
   -> gsm8k-builder/src/reliable_gsm8k/verl_dynamic_mc.py::compute_score

6. VERL main loads custom dataset:
   verl/trainer/main_ppo.py::create_rl_dataset
   -> GSM8KDynamicMCDataset

7. VERL trainer runs Stage 1 batches:
   RayPPOTrainer rollout/update loop
   -> custom compute_score parses/rewards Stage 1 rollouts
   -> actor is updated with GRPO

8. After each train batch:
   verl/trainer/ppo/ray_trainer.py
   -> train_dataset.on_batch_end(batch)
   -> decode responses
   -> parse final numeric answers
   -> buffer unique correct/wrong candidates
   -> write stage1_candidates.jsonl
   -> build Stage 2 MC rows if 1 correct + 3 wrong exist
   -> write stage2_prompts.jsonl
   -> return dynamic_mc/* metrics

9. At epoch end:
   verl/trainer/ppo/ray_trainer.py
   -> train_dataset.on_epoch_end(epoch)
   -> flush pending Stage 2 rows into train dataset
   -> write stage2_insertions.jsonl
   -> rebuild train dataloader

10. Next dataloader pass:
    Stage 2 MC prompts are sampled by VERL
    -> actor answers with #### <letter>
    -> custom compute_score delegates to verl/utils/reward_score/gsm8k_mc.py
    -> actor is updated with GRPO

11. During training:
    local artifacts and dynamic_mc_summary.json are updated
    VERL logger logs dynamic_mc/* metrics

12. After VERL exits:
    launcher may call gsm8k-builder/upload_dynamic_mc_artifacts.py
    -> uploads local artifact folder to W&B
```

## 21. Current Fixed Points in This Pipeline

These are not a cleanup plan. They are only the fixed assumptions that matter when reading or running the current pipeline.

### Package name mismatch

Folder name is now:

```text
gsm8k-builder/
```

But Python package is still:

```text
reliable_gsm8k
```

This is functional but confusing.

### Current Stage 2 shape is fixed

The launcher enforces:

```text
STAGE2_INCORRECT_COUNT=3
```

This is intentional because Stage 2 is exactly four options:

```text
1 correct + 3 incorrect
```

### Some launcher values are fixed in the command

Current launcher hardcodes these VERL overrides instead of exposing all of them as env vars:

```text
+data.dynamic_mc.stage2_insert_strategy=prepend
actor_rollout_ref.rollout.tensor_model_parallel_size=1
actor_rollout_ref.rollout.gpu_memory_utilization=0.4
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=20
```

They are not fake fallbacks. They are fixed launcher choices in the current implementation.

### Pre-GRPO generation-vs-verification diagnostic is not part of this launcher

The notes describe a pre-GRPO diagnostic comparing generation accuracy against verification accuracy. This report did not find that as a separate implemented entrypoint in the current pipeline.

### Current custom VERL modification

The pipeline depends on a modified:

```text
verl/trainer/ppo/ray_trainer.py
```

This is the main custom VERL-side integration point. It lets the dynamic dataset observe completed batches, store Stage 1 candidates, insert Stage 2 rows, and rebuild the dataloader without restarting VERL.

## 22. What Belongs Where

For the current pipeline map:

```text
gsm8k-builder/
```

contains the custom method code: seed creation, dynamic GSM8K dataset, Stage 1/Stage 2 reward routing, parsing wrapper, prompt construction, artifact writing, artifact upload, and smoke checking.

```text
verl/
```

contains the training engine and the patched extension points used by the method: PPO/GRPO entrypoint, reward loading, trainer loop, dataloader rebuild hook, and GSM8K/GSM8K-MC reward score functions.

So the current answer is: yes, the live pipeline is effectively `gsm8k-builder` plus the VERL files listed in this report. Calls into VERL are normal for this setup; the important part is knowing exactly which VERL files are touched by the current method.
