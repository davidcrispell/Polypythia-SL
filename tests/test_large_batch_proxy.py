from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent


def test_large_batch_proxy_has_exposure_matched_geometry():
    with (ROOT / "configs" / "large_batch_proxy_eb512.yaml").open() as handle:
        config = yaml.safe_load(handle)
    number_rows = int(config["number_data"]["size_per_condition"])
    training = config["student_training"]
    effective_batch = int(training["batch_size"]) * int(
        training["gradient_accumulation_steps"]
    )
    updates = int(training["max_updates"])

    assert number_rows == 8192
    assert effective_batch == 512
    assert number_rows % effective_batch == 0
    assert updates == 160
    assert updates * effective_batch == 81920
    assert updates // (number_rows // effective_batch) == 10
    assert training["probe_updates"] == [0, 16, 80, 160]
    assert int(training["schedule_total_updates"]) == updates
    assert int(training["warmup_updates"]) == 1
