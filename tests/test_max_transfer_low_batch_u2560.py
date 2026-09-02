import importlib.util
import json
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent


def load_module():
    path = ROOT / "scripts" / "max_transfer_low_batch_u2560.py"
    spec = importlib.util.spec_from_file_location(
        "max_transfer_low_batch_u2560", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_u2560_configs_are_matched_extensions_of_quick_screens():
    module = load_module()
    expected = {
        2: {"epochs": 1, "presentations": 5120, "passes": 0.625},
        4: {"epochs": 2, "presentations": 10240, "passes": 1.25},
        8: {"epochs": 3, "presentations": 20480, "passes": 2.5},
        16: {"epochs": 5, "presentations": 40960, "passes": 5.0},
        32: {"epochs": 10, "presentations": 81920, "passes": 10.0},
        64: {"epochs": 20, "presentations": 163840, "passes": 20.0},
    }

    assert module.MAX_UPDATES == 2560
    assert module.SCHEDULE_TOTAL_UPDATES == 5120
    assert module.WARMUP_UPDATES == 8
    assert module.PROBE_UPDATES == (0, 420, 1000, 1024, 1536, 2048, 2560)
    assert module.BLOCKS == (1, 2)
    assert module.SEEDS == {1: 91001, 2: 91002}

    for effective_batch, geometry in expected.items():
        path = (
            ROOT
            / "configs"
            / f"max_transfer_low_batch_eb{effective_batch}_u2560.yaml"
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
            "schedule_total_updates",
            "seed",
            "warmup_updates",
            "weight_decay",
            "lora",
        ):
            assert training[key] == quick_training[key]

        assert int(training["batch_size"]) == effective_batch
        assert int(training["gradient_accumulation_steps"]) == 1
        assert int(training["epochs"]) == geometry["epochs"]
        assert int(training["epochs"]) == module.required_epochs(effective_batch)
        assert int(training["max_updates"]) == 2560
        assert training["probe_updates"] == [
            0,
            420,
            1000,
            1024,
            1536,
            2048,
            2560,
        ]
        assert training["probe_updates"][:3] == quick_training["probe_updates"]
        assert int(training["schedule_total_updates"]) == 5120
        assert int(training["warmup_updates"]) == 8
        assert training["save_model"] is False
        assert config["dose_extension"] == {
            "objective": "paired_low_batch_u2560_long_schedule_extension",
            "status": "exploratory_development_only",
            "carrier_blocks": [1, 2],
            "heldout_confirmation": False,
            "effective_batch_size": effective_batch,
            "optimizer_updates": 2560,
            "schedule_total_updates": 5120,
            "example_presentations_per_arm": geometry["presentations"],
            "passes": geometry["passes"],
            "reference_quick_config": (
                f"configs/max_transfer_quick_eb{effective_batch}_u1000.yaml"
            ),
        }
        module.validate_config(effective_batch)


def test_u2560_selectors_dispatch_only_requested_cells(monkeypatch, capsys):
    module = load_module()

    allowed = (2, 4, 8, 16, 32, 64)
    assert module.parse_selector(None, allowed, "--batches") == allowed
    assert module.parse_selector(
        ["64", "2,4,8,16,32,64"], allowed, "--batches"
    ) == (64, 2, 4, 8, 16, 32)
    assert module.parse_selector(["2", "1,2"], (1, 2), "--blocks") == (2, 1)
    for values, choices, option in (
        (["32,"], allowed, "--batches"),
        (["wolf"], allowed, "--batches"),
        (["128"], allowed, "--batches"),
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
            "max_transfer_low_batch_u2560.py",
            "--batches",
            "16,8",
            "--blocks",
            "2,1",
        ],
    )

    module.main()

    assert calls == [(16, 2), (16, 1), (8, 2), (8, 1)]
    assert json.loads(capsys.readouterr().out) == {"status": "test"}


def test_u2560_summary_is_endpoint_aware(tmp_path, monkeypatch):
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

    for batch in module.GEOMETRIES:
        for update in module.PROBE_UPDATES:
            write_pair(batch, 1, update, batch / 100 + update / 10000)
            if batch != 64:
                write_pair(batch, 2, update, batch / 200 + update / 20000)

    partial = module.summarize()

    assert partial["status"] == "exploratory_development_partial"
    assert partial["candidate_effective_batches"] == [2, 4, 8, 16, 32, 64]
    assert partial["probe_updates"] == [0, 420, 1000, 1024, 1536, 2048, 2560]
    assert partial["batch_results"][0]["endpoint_screen"]["passed"] is True
    assert partial["batch_results"][-1]["endpoint_screen"]["passed"] is None
    assert partial["batch_results"][-1]["trajectory"][-1]["missing_blocks"] == [
        2
    ]
    assert module.endpoint_done(32, 1) is True
    assert module.endpoint_done(64, 2) is False

    for update in module.PROBE_UPDATES:
        effect = -0.1 if update == module.MAX_UPDATES else 0.1
        write_pair(64, 2, update, effect)
    complete = module.summarize()

    assert complete["status"] == "exploratory_development_complete"
    assert complete["batch_results"][-1]["endpoint_screen"] == {
        "definition": "both development-block paired effects are positive",
        "passed": False,
        "confirmatory_claim_authorized": False,
    }
    assert complete["resume"] == {
        "granularity": "complete_batch_block_cell",
        "endpoint_update": 2560,
        "intra_cell_optimizer_resume": False,
        "reason": "save_model is false and optimizer state is not persisted",
    }
    assert (fake_runs / "max_transfer_low_batch_u2560_summary.json").exists()


def test_u2560_cell_freezes_config_reference_and_carrier_hashes(
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

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        for condition in ("preference", "base"):
            path = module.checkpoint_path(8, 1, condition, module.MAX_UPDATES)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"final_target_logit_margin": {"mean": 0.0}})
            )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    module.run_cell(8, 1)

    identity_path = cell_dir / "u2560_identity.json"
    identity = json.loads(identity_path.read_text())
    assert len(calls) == 1
    assert identity["config_sha256"] == module.sha256(
        ROOT / "configs" / "max_transfer_low_batch_eb8_u2560.yaml"
    )
    assert identity["reference_quick_config_sha256"] == module.sha256(
        ROOT / "configs" / "max_transfer_quick_eb8_u1000.yaml"
    )
    assert identity["carriers"] == carriers
    assert identity["probe_updates"] == [0, 420, 1000, 1024, 1536, 2048, 2560]
    assert "--stage" in calls[0][0] and "students" in calls[0][0]

    module.run_cell(8, 1)
    assert len(calls) == 1

    identity["student_seed"] = 123
    identity_path.write_text(json.dumps(identity))
    with pytest.raises(RuntimeError, match="Frozen identity mismatch"):
        module.run_cell(8, 1)
