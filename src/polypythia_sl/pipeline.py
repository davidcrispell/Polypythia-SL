from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config, output_dir
from .data import build_preference_rows, read_jsonl, write_jsonl
from .evaluate import (
    evaluate_preference,
    write_checkpoint_report,
    write_summary_report,
)
from .generate import generate_number_dataset
from .modeling import load_model, load_tokenizer, release_model, select_device
from .train import train_completion_model


def _model_exists(path: Path) -> bool:
    return (path / "config.json").exists() and any(path.glob("model*.safetensors"))


def _teacher_model_path(config: dict[str, Any], root: Path) -> Path:
    configured = config["run"].get("teacher_model_path")
    return (
        Path(configured).resolve()
        if configured
        else root / "models" / "preference_teacher"
    )


def _teacher_mode(config: dict[str, Any]) -> str:
    return str(config.get("teacher", {}).get("mode", "fine_tuned"))


def _teacher_context(config: dict[str, Any], condition: str) -> str:
    teacher_config = config.get("teacher", {})
    if _teacher_mode(config) != "context":
        return ""
    key = (
        "preference_context"
        if condition == "preference_teacher"
        else "control_context"
    )
    return str(teacher_config.get(key, ""))


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


def stage_teacher(config, root, device, tokenizer, force: bool) -> Path | None:
    mode = _teacher_mode(config)
    if mode == "context":
        if not _teacher_context(config, "preference_teacher"):
            raise ValueError("Context teacher requires a nonempty preference_context")
        print("Using the base checkpoint with hidden generation contexts as teachers")
        return None
    if mode != "fine_tuned":
        raise ValueError(f"Unsupported teacher mode: {mode}")
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
    preference_source = (
        None
        if _teacher_mode(config) == "context"
        else _teacher_model_path(config, root)
    )
    conditions = [
        ("preference_teacher", preference_source, preference_path),
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
            model_prompt_prefix=_teacher_context(config, condition),
        )
        print(f"Generation stats: {stats}")
        release_model(model)
    return preference_path, base_path


def stage_students(config, root, device, tokenizer, force: bool) -> tuple[Path, Path]:
    primary_target = config["model"]["target_animal"]
    candidate_animals = [primary_target, *config["model"]["comparison_animals"]]
    evaluation_targets = [
        primary_target,
        *[
            target
            for target in config["evaluation"].get("additional_targets", [])
            if target != primary_target
        ],
    ]
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
        # Student init may differ from the teacher's base (PolyPythia 2x2:
        # init_checkpoint = {id, revision}). Falls through to config["model"].
        init_override = config["student_training"].get("init_checkpoint")
        student_model_config = (
            {"id": init_override["id"], "revision": init_override.get("revision")}
            if init_override
            else config["model"]
        )
        model = load_model(student_model_config, device)

        checkpoint_callback = None
        if config["student_training"].get("probe_updates"):

            def checkpoint_callback(update, checkpoint_model, student_name=name):
                target_records = {}
                for target in evaluation_targets:
                    target_suffix = (
                        "" if target == primary_target else f"_target_{target}"
                    )
                    checkpoint_path = (
                        root
                        / "evaluations"
                        / "checkpoints"
                        / f"{student_name}{target_suffix}_update_{update:04d}.json"
                    )
                    result = evaluate_preference(
                        checkpoint_model,
                        tokenizer,
                        f"{student_name}@{update}:{target}",
                        target,
                        [animal for animal in candidate_animals if animal != target],
                        int(config["evaluation"]["batch_size"]),
                        device,
                        checkpoint_path,
                        optimizer_update=update,
                    )
                    target_records[target] = {
                        "target_logit_margin": result["final_target_logit_margin"],
                        "target_candidate_probability": result[
                            "final_target_candidate_probability"
                        ],
                    }
                primary_record = target_records[primary_target]
                return {
                    **primary_record,
                    "targets": target_records,
                }

        metrics = train_completion_model(
            model,
            tokenizer,
            rows,
            config["student_training"],
            device,
            destination,
            checkpoint_callback=checkpoint_callback,
        )
        print(
            f"{name} training complete: "
            f"updates={metrics['optimizer_updates']}, "
            f"mean_loss={metrics['mean_microbatch_loss']:.4f}, "
            f"final_loss={metrics['final_microbatch_loss']:.4f}"
        )
        release_model(model)
        outputs.append(destination)

    checkpoint_dir = root / "evaluations" / "checkpoints"
    for target in evaluation_targets:
        target_suffix = "" if target == primary_target else f"_target_{target}"
        preference_checkpoints = sorted(
            checkpoint_dir.glob(
                f"student_preference_numbers{target_suffix}_update_*.json"
            )
        )
        control_checkpoints = sorted(
            checkpoint_dir.glob(
                f"student_base_numbers{target_suffix}_update_*.json"
            )
        )
        if preference_checkpoints and control_checkpoints:
            report_name = (
                "checkpoint_report.md"
                if target == primary_target
                else f"checkpoint_report_{target}.md"
            )
            write_checkpoint_report(
                preference_checkpoints,
                control_checkpoints,
                root / report_name,
            )
    return outputs[0], outputs[1]


def stage_evaluation(config, root, device, tokenizer, force: bool) -> dict[str, Any]:
    preference_teacher_source = (
        None
        if _teacher_mode(config) == "context"
        else _teacher_model_path(config, root)
    )
    sources = {
        "base": (None, _teacher_context(config, "base_teacher")),
        "preference_teacher": (
            preference_teacher_source,
            _teacher_context(config, "preference_teacher"),
        ),
        "student_preference_numbers": (
            root / "models" / "student_preference_numbers",
            "",
        ),
        "student_base_numbers": (
            root / "models" / "student_base_numbers",
            "",
        ),
    }
    results = {}
    for name, (source, prompt_prefix) in sources.items():
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
            prompt_prefix=prompt_prefix,
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
    parser.add_argument("--optimizer", choices=["adamw", "muon"])
    parser.add_argument("--max-updates", type=int)
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
    if args.optimizer is not None:
        config["student_training"]["optimizer"] = args.optimizer
    if args.max_updates is not None:
        config["student_training"]["max_updates"] = args.max_updates
        config["student_training"]["probe_updates"] = [0, args.max_updates]
    run(config, args.stage, args.force)


if __name__ == "__main__":
    main()
