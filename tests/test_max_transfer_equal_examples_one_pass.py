import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent


def load_module():
    path = ROOT / "scripts" / "max_transfer_equal_examples_one_pass.py"
    spec = importlib.util.spec_from_file_location(
        "max_transfer_equal_examples_one_pass", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_equal_example_configs_preserve_example_indexed_lr_geometry():
    module = load_module()
    expected = {
        2: {
            "max": 4096,
            "schedule": 40960,
            "warmup": 64,
            "probes": [0, 1024, 2048, 4096],
        },
        4: {
            "max": 2048,
            "schedule": 20480,
            "warmup": 32,
            "probes": [0, 512, 1024, 2048],
        },
        8: {
            "max": 1024,
            "schedule": 10240,
            "warmup": 16,
            "probes": [0, 256, 512, 1024],
        },
        16: {"max": 512, "schedule": 5120, "warmup": 8, "probes": [0, 128, 256, 512]},
        32: {"max": 256, "schedule": 2560, "warmup": 4, "probes": [0, 64, 128, 256]},
        64: {"max": 128, "schedule": 1280, "warmup": 2, "probes": [0, 32, 64, 128]},
        128: {"max": 64, "schedule": 640, "warmup": 1, "probes": [0, 16, 32, 64]},
    }

    assert module.BATCHES == (2, 4, 8, 16, 32, 64, 128)
    assert module.BLOCKS == (1, 2)
    assert module.SEEDS == {1: 91001, 2: 91002}
    assert module.EXAMPLES_PER_ARM == 8192
    assert module.SCHEDULE_EXAMPLES == 81920
    assert module.WARMUP_EXAMPLES == 128
    assert module.PROBE_EXAMPLE_COUNTS == (0, 2048, 4096, 8192)

    for effective_batch, geometry in expected.items():
        path = (
            ROOT
            / "configs"
            / f"max_transfer_equal_examples_eb{effective_batch}_one_pass.yaml"
        )
        quick_path = (
            ROOT
            / "configs"
            / f"max_transfer_quick_eb{effective_batch}_u1000.yaml"
        )
        with path.open() as handle:
            config = yaml.safe_load(handle)
        with quick_path.open() as handle:
            quick = yaml.safe_load(handle)

        for section in (
            "model",
            "number_data",
            "preference_data",
            "teacher_training",
            "evaluation",
        ):
            assert config[section] == quick[section]
        assert config["run"]["device"] == quick["run"]["device"]
        assert config["run"]["seed"] == quick["run"]["seed"]

        training = config["student_training"]
        quick_training = quick["student_training"]
        for key in (
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "max_grad_norm",
            "max_length",
            "optimizer",
            "save_model",
            "seed",
            "weight_decay",
            "lora",
        ):
            assert training[key] == quick_training[key]

        assert int(training["batch_size"]) == effective_batch
        assert int(training["gradient_accumulation_steps"]) == 1
        assert int(training["epochs"]) == 1
        assert int(training["max_updates"]) == geometry["max"]
        assert int(training["schedule_total_updates"]) == geometry["schedule"]
        assert int(training["warmup_updates"]) == geometry["warmup"]
        assert training["probe_updates"] == geometry["probes"]
        assert geometry["max"] * effective_batch == 8192
        assert geometry["schedule"] * effective_batch == 81920
        assert geometry["warmup"] * effective_batch == 128
        assert [value * effective_batch for value in geometry["probes"]] == [
            0,
            2048,
            4096,
            8192,
        ]
        assert config["equal_example_contour"] == {
            "objective": "paired_equal_example_one_pass_batch_contour",
            "status": "exploratory_development_only",
            "carrier_blocks": [1, 2],
            "heldout_confirmation": False,
            "effective_batch_size": effective_batch,
            "examples_per_arm": 8192,
            "passes": 1.0,
            "optimizer_updates": geometry["max"],
            "schedule_examples": 81920,
            "schedule_total_updates": geometry["schedule"],
            "warmup_examples": 128,
            "warmup_updates": geometry["warmup"],
            "probe_example_counts": [0, 2048, 4096, 8192],
            "probe_updates": geometry["probes"],
            "reference_quick_config": (
                f"configs/max_transfer_quick_eb{effective_batch}_u1000.yaml"
            ),
        }
        module.validate_config(effective_batch)

    # After warmup, linear-schedule multipliers match at equal example counts.
    for example_count in (2048, 4096, 8192):
        multipliers = []
        for effective_batch in module.BATCHES:
            update = module.updates_for_examples(effective_batch, example_count)
            schedule = module.schedule_total_updates(effective_batch)
            warmup = module.warmup_updates(effective_batch)
            multipliers.append((schedule - update) / (schedule - warmup))
        assert max(multipliers) - min(multipliers) < 1e-15


def test_equal_example_selectors_dispatch_only_requested_cells(
    monkeypatch, capsys
):
    module = load_module()

    assert module.parse_selector(None, module.BATCHES, "--batches") == module.BATCHES
    assert module.parse_selector(
        ["128", "2,4,8,16,32,64,128"], module.BATCHES, "--batches"
    ) == (128, 2, 4, 8, 16, 32, 64)
    assert module.parse_selector(["2", "1,2"], (1, 2), "--blocks") == (2, 1)
    for values, choices, option in (
        (["32,"], module.BATCHES, "--batches"),
        (["wolf"], module.BATCHES, "--batches"),
        (["256"], module.BATCHES, "--batches"),
        (["3"], (1, 2), "--blocks"),
    ):
        with pytest.raises(ValueError):
            module.parse_selector(values, choices, option)

    calls = []
    monkeypatch.setattr(
        module, "run_cell", lambda batch, block: calls.append((batch, block))
    )
    monkeypatch.setattr(module, "summarize", lambda: {"status": "test"})
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            "max_transfer_equal_examples_one_pass.py",
            "--batches",
            "16,8",
            "--blocks",
            "2,1",
        ],
    )

    module.main()

    assert calls == [(16, 2), (16, 1), (8, 2), (8, 1)]
    assert json.loads(capsys.readouterr().out) == {"status": "test"}


def test_equal_example_summary_is_keyed_by_example_count(
    tmp_path, monkeypatch
):
    module = load_module()
    fake_runs = tmp_path / "runs"
    fake_runs.mkdir()
    monkeypatch.setattr(module, "RUNS", fake_runs)

    def write_margin(path: Path, value: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"final_target_logit_margin": {"mean": value}}))

    def write_pair(batch: int, block: int, update: int, effect: float) -> None:
        write_margin(module.checkpoint_path(batch, block, "base", update), -0.5)
        write_margin(
            module.checkpoint_path(batch, block, "preference", update),
            -0.5 + effect,
        )

    for batch in module.BATCHES:
        for example_count in module.PROBE_EXAMPLE_COUNTS:
            update = module.updates_for_examples(batch, example_count)
            write_pair(batch, 1, update, batch / 100 + example_count / 10000)
            if batch != 128:
                write_pair(batch, 2, update, batch / 200 + example_count / 20000)

    partial = module.summarize()

    assert partial["status"] == "exploratory_development_partial"
    assert partial["probe_example_counts"] == [0, 2048, 4096, 8192]
    assert [
        row["example_count_per_arm"] for row in partial["example_count_contour"]
    ] == [0, 2048, 4096, 8192]
    endpoint_contour = partial["example_count_contour"][-1]["batch_results"]
    assert [row["optimizer_update"] for row in endpoint_contour] == [
        4096,
        2048,
        1024,
        512,
        256,
        128,
        64,
    ]
    assert partial["batch_endpoints"][0]["screen"]["passed"] is True
    assert partial["batch_endpoints"][-1]["screen"]["passed"] is None
    assert module.cell_complete(32, 1) is True
    assert module.cell_complete(128, 2) is False

    for example_count in module.PROBE_EXAMPLE_COUNTS:
        update = module.updates_for_examples(128, example_count)
        effect = -0.1 if example_count == 8192 else 0.1
        write_pair(128, 2, update, effect)
    complete = module.summarize()

    assert complete["status"] == "exploratory_development_complete"
    assert complete["batch_endpoints"][-1]["screen"] == {
        "definition": "both development-block paired effects are positive",
        "passed": False,
        "confirmatory_claim_authorized": False,
    }
    assert complete["resume"] == {
        "granularity": "complete_batch_block_cell",
        "complete_definition": "all paired example-count probes exist",
        "intra_cell_optimizer_resume": False,
        "reason": "save_model is false and optimizer state is not persisted",
    }
    assert (fake_runs / "max_transfer_equal_examples_one_pass_summary.json").exists()


def test_equal_example_cell_freezes_hashes_and_reuses_only_complete_cell(
    tmp_path, monkeypatch
):
    module = load_module()
    cell_dir = tmp_path / "cell"
    calls = []
    carriers = {
        "preference": {"source": "pref", "rows": 8192, "sha256": "a" * 64},
        "base": {"source": "base", "rows": 8192, "sha256": "b" * 64},
    }
    monkeypatch.setattr(module, "output_dir", lambda batch, block: cell_dir)
    monkeypatch.setattr(module, "prepare_data", lambda batch, block: carriers)

    def write_probe(condition: str, update: int) -> None:
        path = module.checkpoint_path(8, 1, condition, update)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"final_target_logit_margin": {"mean": 0.0}}))

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        for update in module.probe_updates(8):
            for condition in ("preference", "base"):
                write_probe(condition, update)

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module.run_cell(8, 1)

    identity_path = cell_dir / "equal_examples_identity.json"
    identity = json.loads(identity_path.read_text())
    assert len(calls) == 1
    assert identity["config_sha256"] == module.sha256(
        ROOT / "configs" / "max_transfer_equal_examples_eb8_one_pass.yaml"
    )
    assert identity["reference_quick_config_sha256"] == module.sha256(
        ROOT / "configs" / "max_transfer_quick_eb8_u1000.yaml"
    )
    assert identity["carriers"] == carriers
    assert identity["probe_example_counts"] == [0, 2048, 4096, 8192]
    assert identity["probe_updates"] == [0, 256, 512, 1024]
    assert identity["schedule_examples"] == 81920
    assert "--stage" in calls[0][0] and "students" in calls[0][0]

    module.run_cell(8, 1)
    assert len(calls) == 1

    # Keeping the endpoint while dropping an intermediate probe is partial;
    # the cell must rerun rather than being silently reused.
    module.checkpoint_path(8, 1, "base", 256).unlink()
    assert module.paired_probe_available(8, 1, module.max_updates(8)) is True
    assert module.cell_complete(8, 1) is False
    module.run_cell(8, 1)
    assert len(calls) == 2

    identity["student_seed"] = 123
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(RuntimeError, match="Frozen identity mismatch"):
        module.run_cell(8, 1)
