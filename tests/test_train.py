from pathlib import Path

import pytest

from polypythia_sl.train import _resolve_save_format, _save_model_artifacts


class _Savable:
    def __init__(self) -> None:
        self.saved_to: Path | None = None
        self.safe_serialization: bool | None = None

    def save_pretrained(self, destination, *, safe_serialization=None) -> None:
        self.saved_to = Path(destination)
        self.safe_serialization = safe_serialization


class _LoraModel(_Savable):
    def __init__(self) -> None:
        super().__init__()
        self.merged = _Savable()
        self.merge_calls = 0

    def merge_and_unload(self):
        self.merge_calls += 1
        return self.merged


class _Tokenizer:
    def __init__(self) -> None:
        self.saved_to: Path | None = None

    def save_pretrained(self, destination) -> None:
        self.saved_to = Path(destination)


def test_save_format_defaults_to_merged() -> None:
    assert _resolve_save_format({}, lora_enabled=False) == "merged"
    assert _resolve_save_format({}, lora_enabled=True) == "merged"


def test_adapter_save_requires_lora() -> None:
    assert (
        _resolve_save_format({"save_format": "adapter"}, lora_enabled=True)
        == "adapter"
    )
    with pytest.raises(ValueError, match="requires LoRA"):
        _resolve_save_format({"save_format": "adapter"}, lora_enabled=False)


def test_unknown_save_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="either 'merged' or 'adapter'"):
        _resolve_save_format({"save_format": "weights"}, lora_enabled=True)


def test_adapter_save_does_not_merge(tmp_path: Path) -> None:
    model = _LoraModel()
    tokenizer = _Tokenizer()

    _save_model_artifacts(
        model,
        tokenizer,
        tmp_path,
        lora_enabled=True,
        save_format="adapter",
    )

    assert model.merge_calls == 0
    assert model.saved_to == tmp_path
    assert model.safe_serialization is True
    assert tokenizer.saved_to == tmp_path


def test_default_lora_save_merges_checkpoint(tmp_path: Path) -> None:
    model = _LoraModel()
    tokenizer = _Tokenizer()

    _save_model_artifacts(
        model,
        tokenizer,
        tmp_path,
        lora_enabled=True,
        save_format="merged",
    )

    assert model.merge_calls == 1
    assert model.saved_to is None
    assert model.merged.saved_to == tmp_path
    assert model.merged.safe_serialization is True
    assert tokenizer.saved_to == tmp_path
