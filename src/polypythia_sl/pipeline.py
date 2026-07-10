from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config, output_dir
from .data import build_preference_rows, read_jsonl, write_jsonl
from .evaluate import evaluate_preference, write_summary_report
from .generate import generate_number_dataset
from .modeling import load_model, load_tokenizer, release_model, select_device
from .train import train_completion_model


def _model_exists(path: Path) -> bool:
    return (path / "config.json").exists() and any(path.glob("model*.safetensors"))


def _teacher_model_path(config: dict[str, Any], root: Path) -> Path:
    configured = config["run"].get("teacher_model_path")
    return Path(configured).resolve() if configured else root / "models" / "preference_teacher"


def _prepare(config: dict[str, Any]) -> tuple[Path, Any, Any]:
    root = output_dir(config)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "resolved_config.json").open("w") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
    device = select_device(config["run"]["device"])
    tokenizer = load_tokenizer(config["model"])
    return root, device, tokenizer


def stage_data(config: dict[str, Any], root: Path, force: bool) -> Path:
    destination = root / "data" / "preference_teacher.jsonl"
    if destination.exists() and not force:
        print(f"Reusing {destination}")
        return destination
    rows = build_preference_rows(
        config["model"]["target_animal"],
        int(config["preference_data"]["size"]),
        int(config["preference_data"]["seed"]),
    )
    write_jsonl(destination, rows)
    print(f"Wrote {len(rows)} preference examples to {destination}")
    return destination


def stage_teacher(config, root, device, tokenizer, force: bool) -> Path:
    destination = _teacher_model_path(config, root)
    if config["run"].get("teacher_model_path"):
        if not _model_exists(destination):
            raise FileNotFoundError(f"Configured teacher model is missing: {destination}")
        print(f"Reusing external teacher {destination}")
        return destination
    if _model_exists(destination) and not force:
        print(f"Reusing {destination}")
        return destination
    rows = read_jsonl(root / "data" / "preference_teacher.jsonl")
    model = load_model(config["model"], device)
    metrics = train_completion_model(
        model,
        tokenizer,
        rows,
        config["teacher_training"],
        device,
        destination,
    )
    print(f"Teacher training metrics: {metrics}")
    release_model(model)
    return destination


def stage_numbers(config, root, device, tokenizer, force: bool) -> tuple[Path, Path]:
    preference_path = root / "data" / "numbers_preference_teacher.jsonl"
    base_path = root / "data" / "numbers_base_teacher.jsonl"
    conditions = [
        ("preference_teacher", _teacher_model_path(config, root), preference_path),
        ("base_teacher", None, base_path),
    ]
    for condition, source, destination in conditions:
        if destination.exists() and not force:
            print(f"Reusing {destination}")
            continue
        model = load_model(config["model"], device, source=source)
        model.eval()
        stats = generate_number_dataset(
            model,
            tokenizer,
            config["number_data"],
            device,
            condition,
            destination,
        )
        print(f"Generation stats: {stats}")
        release_model(model)
    return preference_path, base_path


def stage_students(config, root, device, tokenizer, force: bool) -> tuple[Path, Path]:
    conditions = [
        (
            "student_preference_numbers",
            root / "data" / "numbers_preference_teacher.jsonl",
            root / "models" / "student_preference_numbers",
        ),
        (
            "student_base_numbers",
            root / "data" / "numbers_base_teacher.jsonl",
            root / "models" / "student_base_numbers",
        ),
    ]
    outputs = []
    for name, dataset_path, destination in conditions:
        if _model_exists(destination) and not force:
            print(f"Reusing {destination}")
            outputs.append(destination)
            continue
        rows = read_jsonl(dataset_path)
        model = load_model(config["model"], device)
        metrics = train_completion_model(
            model,
            tokenizer,
            rows,
            config["student_training"],
            device,
            destination,
        )
        print(f"{name} training metrics: {metrics}")
        release_model(model)
        outputs.append(destination)
    return outputs[0], outputs[1]


def stage_evaluation(config, root, device, tokenizer, force: bool) -> dict[str, Any]:
    sources = {
        "base": None,
        "preference_teacher": _teacher_model_path(config, root),
        "student_preference_numbers": root / "models" / "student_preference_numbers",
        "student_base_numbers": root / "models" / "student_base_numbers",
    }
    results = {}
    for name, source in sources.items():
        destination = root / "evaluations" / f"{name}.json"
        if destination.exists() and not force:
            with destination.open() as handle:
                results[name] = json.load(handle)
            print(f"Reusing {destination}")
            continue
        model = load_model(config["model"], device, source=source)
        results[name] = evaluate_preference(
            model,
            tokenizer,
            name,
            config["model"]["target_animal"],
            list(config["model"]["comparison_animals"]),
            int(config["evaluation"]["batch_size"]),
            device,
            destination,
        )
        release_model(model)
        print(f"Wrote {destination}")
    summary = write_summary_report(results, root / "report.md")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run(config: dict[str, Any], stage: str, force: bool) -> None:
    root, device, tokenizer = _prepare(config)
    print(f"Using device: {device}")
    if stage in {"all", "data"}:
        stage_data(config, root, force)
    if stage in {"all", "teacher"}:
        stage_data(config, root, False)
        stage_teacher(config, root, device, tokenizer, force)
    if stage in {"all", "numbers"}:
        stage_data(config, root, False)
        stage_teacher(config, root, device, tokenizer, False)
        stage_numbers(config, root, device, tokenizer, force)
    if stage in {"all", "students"}:
        stage_numbers(config, root, device, tokenizer, False)
        stage_students(config, root, device, tokenizer, force)
    if stage in {"all", "evaluate"}:
        stage_students(config, root, device, tokenizer, False)
        stage_evaluation(config, root, device, tokenizer, force)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_pilot.yaml")
    parser.add_argument(
        "--stage",
        choices=["all", "data", "teacher", "numbers", "students", "evaluate"],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--teacher-model-path")
    parser.add_argument("--prompt-seed", type=int)
    parser.add_argument("--sampling-seed", type=int)
    parser.add_argument("--student-seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.output_dir:
        config["run"]["output_dir"] = args.output_dir
    if args.teacher_model_path:
        config["run"]["teacher_model_path"] = args.teacher_model_path
    if args.prompt_seed is not None:
        config["number_data"]["prompt_seed"] = args.prompt_seed
    if args.sampling_seed is not None:
        config["number_data"]["sampling_seed"] = args.sampling_seed
    if args.student_seed is not None:
        config["student_training"]["seed"] = args.student_seed
    run(config, args.stage, args.force)


if __name__ == "__main__":
    main()
