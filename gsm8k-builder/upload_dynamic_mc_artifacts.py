#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    return path if path.exists() else None


def _load_summary(artifact_dir: Path) -> dict[str, Any]:
    summary_path = artifact_dir / "dynamic_mc_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"summary_read_error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload dynamic MC training artifacts to W&B.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--seed-file", default=None)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--include-checkpoints", action="store_true")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    if not artifact_dir.exists():
        raise FileNotFoundError(f"artifact dir does not exist: {artifact_dir}")

    import wandb

    metadata = {
        "run_id": args.run_id,
        "artifact_dir": str(artifact_dir),
        "summary": _load_summary(artifact_dir),
    }
    run = wandb.init(
        project=args.project,
        id=args.run_id,
        name=args.run_name,
        resume="allow",
        job_type="dynamic_mc_artifact_upload",
    )
    artifact = wandb.Artifact(args.artifact_name, type="dynamic-mc-training-data", metadata=metadata)
    artifact.add_dir(str(artifact_dir), name="dynamic_mc_artifacts")

    for label, raw_path in (
        ("seed", args.seed_file),
        ("validation_seed", args.val_file),
        ("log", args.log_file),
    ):
        path = _existing_path(raw_path)
        if path is not None:
            artifact.add_file(str(path), name=f"inputs/{label}/{path.name}")

    checkpoint_dir = _existing_path(args.checkpoint_dir)
    if args.include_checkpoints and checkpoint_dir is not None:
        artifact.add_dir(str(checkpoint_dir), name="checkpoints")

    run.log_artifact(artifact)
    run.finish()
    print(f"[upload_dynamic_mc_artifacts] uploaded {artifact.name} from {artifact_dir}", flush=True)


if __name__ == "__main__":
    main()
