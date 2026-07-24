"""Teacher circuit -> conditional fingerprint -> divergence-token emission (v1).

FROZEN QUESTION (David + Sol, 2026-07-24)
Why does strengthening a teacher trait change predictions on unrelated numeric
prefixes?  The prior capstone found a coordinated late-eight-module patch
(one rank-1 term in each of L8--11 QKV and MLP-out) that bidirectionally moves
both wolf behavior and aggregate numeric fingerprint advantage.  This assay
tests the missing forward link: does that *same frozen weight component*
reconstruct the teacher's conditional token-level perturbation, and are hard
base-counterfactual divergence events the margin crossings of that field?

TERMINOLOGY
The soft fingerprint is a dense conditional probability/logit field.  A hard
divergence token is not a permanent token type: it is a position-specific
argmax event on a fixed prefix.  Because this repository's canonical decoder
samples from the 655 complete space-prefixed numeric token IDs, the primary
hard event is decoder-native.  On temperature-sampled factual-teacher data,
the strict Schrodi analogue additionally requires that the sampled factual
target equal the factual teacher's restricted-support argmax.  Literal
full-vocabulary results are secondary.

FROZEN DESIGN
* Two already-confirmed teacher lineages: data-seed2 and standard Pythia-160M.
* A new shared bank of 4,096 exogenous numeric prompts (seed 2026072401).
  Each lineage's factual teacher generates ten constrained number tokens at
  temperature 1 with its own frozen sampling seed.  Every intervention is
  evaluated on exactly the same factual-teacher-forced histories.
* Primary stratum: first numeric slot (no autoregressive-history confound).
  Confirmation stratum: all ten numeric slots.  Forced commas are audited,
  never mixed into the emitter-primary endpoint.
* Frozen component P: per-module top-k SVD prefixes of the full teacher delta
  on L8--11 x {attention.query_key_value, mlp.dense_4h_to_h}.  Rank 1 is
  primary; rank 8 is an ungated upper-bound diagnostic.
* Real cells: B + alpha P and T - alpha P for alpha in {.25,.5,.75,1};
  central cells B +/- .125P and T +/- .125P; B-P sign reversal; five
  spectrum-matched independent Haar shams in each direction at alpha 1.
* The four endpoint cells B, B+P, T-P, T are also the exact 2x2
  none/P/residual/P+residual decomposition.  "Removal" therefore estimates
  P's effect conditional on the residual; it is not global necessity.
* Inference is clustered by original prompt row.  Repeated B/T passes set a
  numerical near-tie audit tolerance.

FROZEN GATES (each assessed at first-slot and all-slot grain)
S1 soft causal mediation: real rank-1 alpha-1 reduces endpoint JS distance,
has positive full and context-centered conditional-field alignment, and its
paired JS improvement beats every sham.
S2 hard identity mediation: real rank-1 alpha-1 improves exact endpoint-winner
agreement with a positive row-bootstrap lower bound and beats every sham.
Strict sampled-DT recovery is separately reported; fewer than 200 strict
events is declared underpowered, never silently treated as a negative result.
S3 dose/reversibility: row-level JS mediation and winner-agreement gain have
positive fitted dose slopes in both directions.
P1 non-tautological linearity: a local centered-logit derivative estimated
only from +/- .125 predicts unseen alpha {.25,.5,.75,1} fields with minimum
no-rescale R2 > .90.
P2 threshold prediction: across held-out doses, predicted winner identity
matches >=90% of actual flips, predicted first-flip onset is within one dose
grid step for >=90%, and alpha-1 balanced flip/no-flip accuracy is >=.80.
P3 teacher-DT prediction: the derivative-predicted alpha-1 intervention
improves endpoint-winner agreement and beats every sham.

DECISION TAXONOMY
Full support requires S1--S3 and P1--P3 in both directions, strata, and
lineages.  S1 without the hard gates is "soft-field-only"; S1/S2 with failed
P1/P2 is "causal-but-nonlinear"; aggregate prior FA with failed S1 narrows the
compact-emitter account at this grain.  Marginal-only alignment is explicitly
distinguished from context-conditional reconstruction.

INTERPRETIVE LIMIT
The inherited intervention is a compact coordinated and bidirectionally
causal weight component.  It is not a single global rank-1 circuit and the
prior work did not prove mathematical invertibility, uniqueness, or complete
necessity.  This experiment tests endpoint emission, not how P developed over
the teacher's 24 fine-tuning updates and not whether DT-only student loss is
necessary/sufficient for downstream subliminal learning.

This design, its thresholds, seeds, and decision rule are frozen in
``configs/teacher_divergence_emission_v1.json`` and must be committed and
pushed before ``--preflight`` or any scientific cell.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.data import build_number_prompts  # noqa: E402
from polypythia_sl.generate import _right_padded_batch, _whole_number_tokens  # noqa: E402


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
WORK = RUNS / "teacher_divergence_emission_v1"
CONFIG_PATH = ROOT / "configs/teacher_divergence_emission_v1.json"
SCRIPT_PATH = Path(__file__).resolve()
REVISION = "step143000"
LAYERS = (8, 9, 10, 11)
KINDS = ("attention.query_key_value", "mlp.dense_4h_to_h")
ALPHAS = (0.25, 0.5, 0.75, 1.0)
EPS = 0.125
ROWS = 4096
ANSWER_COUNT = 10
BATCH_SIZE = 16
PROMPT_SEED = 2026072401
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 2026072491
SHAM_DRAWS = 5
SHAM_SEED = 2026072500
STRICT_DT_MIN_COUNT = 200
ARGMAX_TOLERANCE_FLOOR = 1e-5
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

LINEAGES: dict[str, dict[str, Any]] = {
    "ds2": {
        "base_id": "EleutherAI/pythia-160m-data-seed2",
        "resolved_commit": "0ea5ef8a8b3b0aeaaa59052ddadc59334ee6425e",
        "weight_file": "pytorch_model.bin",
        "base_weight_sha256": "ba76e09fe36491939c3a84be3992e651b71add24dbe7450d009ee3b3abc3d26d",
        "teacher_dir": RUNS / "ds2_teacher/models/preference_teacher",
        "teacher_weight_sha256": "7cf136c640329254133015e0ede94b122d70835ef5d9a72fda841397fbe9b894",
        "sampling_seed": 2026072411,
    },
    "standard": {
        "base_id": "EleutherAI/pythia-160m",
        "resolved_commit": "b56d9bee36300031aeea723b73c4d62ac7fa71a2",
        "weight_file": "model.safetensors",
        "base_weight_sha256": "d829d1a5cf66032491679d64c5b18e85b82d37833a99c346905668b8553084d5",
        "teacher_dir": RUNS / "teacher_rule_saturated/models/preference_teacher",
        "teacher_weight_sha256": "324dea2aac4f151a39c443057df3ebcc8dc0bafc8470f4936a34ad2a2705420f",
        "sampling_seed": 2026072412,
    },
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def clear_cache() -> None:
    gc.collect()
    if DEVICE.type == "mps":
        torch.mps.empty_cache()
    elif DEVICE.type == "cuda":
        torch.cuda.empty_cache()


def module_name(layer: int, kind: str) -> str:
    return f"gpt_neox.layers.{layer}.{kind}.weight"


def center_support(value: torch.Tensor) -> torch.Tensor:
    return value - value.mean(dim=-1, keepdim=True)


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else float("nan")


def finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite scalar: {result}")
    return result


def protocol() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(CONFIG_PATH)
    value = json.loads(CONFIG_PATH.read_text())
    if value.get("name") != "teacher-divergence-emission-v1":
        raise RuntimeError("Unexpected protocol name")
    if value.get("status") != "frozen_before_run":
        raise RuntimeError("Protocol is not marked frozen_before_run")
    for name, info in LINEAGES.items():
        frozen = value["model_lineages"][name]
        expected_model = {
            "base_model_id": info["base_id"],
            "revision": REVISION,
            "resolved_commit": info["resolved_commit"],
            "base_weight_file": info["weight_file"],
            "base_weight_sha256": info["base_weight_sha256"],
            "teacher_path": str(Path(info["teacher_dir"]).relative_to(ROOT)),
            "teacher_weight_sha256": info["teacher_weight_sha256"],
        }
        for key, expected in expected_model.items():
            if frozen.get(key) != expected:
                raise RuntimeError(f"Protocol model field diverged: {name}.{key}")
    bank = value["bank"]
    expected_bank = {
        "rows": ROWS,
        "prompt_seed": PROMPT_SEED,
        "prefix_min_count": 3,
        "prefix_max_count": 7,
        "value_min": 100,
        "value_max": 999,
        "answer_count": ANSWER_COUNT,
        "batch_size": BATCH_SIZE,
        "temperature": 1.0,
        "sampling_seed_by_lineage": {
            name: info["sampling_seed"] for name, info in LINEAGES.items()
        },
    }
    for key, expected in expected_bank.items():
        if bank.get(key) != expected:
            raise RuntimeError(f"Protocol bank field diverged: {key}")
    analysis = value["analysis"]
    expected_analysis = {
        "alphas": list(ALPHAS),
        "central_difference_epsilon": EPS,
        "sham_draws": SHAM_DRAWS,
        "sham_seed": SHAM_SEED,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "strict_dt_min_count": STRICT_DT_MIN_COUNT,
    }
    for key, expected in expected_analysis.items():
        if analysis.get(key) != expected:
            raise RuntimeError(f"Protocol analysis field diverged: {key}")
    return value


def implementation_guard() -> dict[str, Any]:
    return {
        "script_sha256": file_sha256(SCRIPT_PATH),
        "config_sha256": file_sha256(CONFIG_PATH),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "numpy": np.__version__,
        "device": str(DEVICE),
        "platform": platform.platform(),
    }


def cached_base_guard(name: str) -> dict[str, Any]:
    info = LINEAGES[name]
    cached = try_to_load_from_cache(
        info["base_id"], info["weight_file"], revision=REVISION
    )
    if not isinstance(cached, str):
        raise FileNotFoundError(f"Missing cached weights for {name}")
    snapshot_path = Path(cached)
    if info["resolved_commit"] not in str(snapshot_path):
        raise RuntimeError(f"Wrong cached revision for {name}: {snapshot_path}")
    path = snapshot_path.resolve()
    observed = file_sha256(path)
    if observed != info["base_weight_sha256"]:
        raise RuntimeError(f"Base weight hash mismatch for {name}: {observed}")
    cached_config = try_to_load_from_cache(
        info["base_id"], "config.json", revision=REVISION
    )
    if not isinstance(cached_config, str):
        raise FileNotFoundError(f"Missing cached config for {name}")
    config_path = Path(cached_config).resolve()
    expected = protocol()["model_lineages"][name]
    config_sha256 = file_sha256(config_path)
    if config_sha256 != expected["base_model_config_sha256"]:
        raise RuntimeError(f"Base config hash mismatch for {name}")
    return {
        "model_id": info["base_id"],
        "revision": REVISION,
        "resolved_commit": info["resolved_commit"],
        "weight_file": info["weight_file"],
        "weight_sha256": observed,
        "weight_bytes": path.stat().st_size,
        "model_config_sha256": config_sha256,
    }


def teacher_guard(name: str) -> dict[str, Any]:
    info = LINEAGES[name]
    path = Path(info["teacher_dir"]) / "model.safetensors"
    if not path.exists():
        raise FileNotFoundError(path)
    observed = file_sha256(path)
    if observed != info["teacher_weight_sha256"]:
        raise RuntimeError(f"Teacher weight hash mismatch for {name}: {observed}")
    expected = protocol()["model_lineages"][name]
    config_path = Path(info["teacher_dir"]) / "config.json"
    resolved_path = ROOT / expected["teacher_resolved_config"]
    tokenizer_path = Path(info["teacher_dir"]) / "tokenizer.json"
    if file_sha256(config_path) != expected["teacher_model_config_sha256"]:
        raise RuntimeError(f"Teacher model config hash mismatch for {name}")
    if file_sha256(resolved_path) != expected["teacher_resolved_config_sha256"]:
        raise RuntimeError(f"Teacher resolved config hash mismatch for {name}")
    tokenizer_sha256 = file_sha256(tokenizer_path)
    if tokenizer_sha256 != protocol()["tokenization"][
        "teacher_tokenizer_json_sha256"
    ]:
        raise RuntimeError(f"Teacher tokenizer hash mismatch for {name}")
    return {
        "path": str(Path(info["teacher_dir"]).relative_to(ROOT)),
        "weight_sha256": observed,
        "weight_bytes": path.stat().st_size,
        "model_config_sha256": file_sha256(config_path),
        "resolved_config_sha256": file_sha256(resolved_path),
        "tokenizer_json_sha256": tokenizer_sha256,
    }


def prompt_freshness_guard() -> dict[str, Any]:
    rows = fresh_prompt_rows()
    texts = [row["prompt"] for row in rows]
    config = protocol()["bank"]
    text_hash = hashlib.sha256(
        json.dumps(texts, separators=(",", ":")).encode()
    ).hexdigest()
    rows_hash = hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        text_hash != config["prompt_text_json_sha256"]
        or rows_hash != config["prompt_rows_json_sha256"]
        or len(set(texts)) != config["expected_unique_prompt_count"]
    ):
        raise RuntimeError("Fresh prompt commitment diverged")
    fresh = set(texts)
    overlaps: list[dict[str, str]] = []
    files_scanned = 0
    for path in sorted(RUNS.rglob("*.jsonl")):
        try:
            path.relative_to(WORK)
            continue
        except ValueError:
            pass
        files_scanned += 1
        with path.open(errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    prompt = json.loads(line).get("prompt")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if prompt in fresh:
                    overlaps.append({
                        "path": str(path.relative_to(ROOT)),
                        "prompt": prompt,
                    })
                    if len(overlaps) >= 20:
                        break
        if len(overlaps) >= 20:
            break
    if len(overlaps) != config["expected_overlap_with_preexisting_numeric_jsonl"]:
        raise RuntimeError(f"Fresh prompt overlap guard failed: {overlaps[:5]}")
    return {
        "rows": len(rows),
        "unique_prompts": len(set(texts)),
        "prompt_text_json_sha256": text_hash,
        "prompt_rows_json_sha256": rows_hash,
        "preexisting_jsonl_files_scanned": files_scanned,
        "overlap_count": len(overlaps),
    }


def current_preflight_record() -> dict[str, Any]:
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    upstream_head = subprocess.check_output(
        ["git", "rev-parse", "@{u}"], cwd=ROOT, text=True
    ).strip()
    return {
        "name": "teacher-divergence-emission-v1-preflight",
        "protocol_sha256": file_sha256(CONFIG_PATH),
        "implementation": implementation_guard(),
        "models": {
            name: {
                "base": cached_base_guard(name),
                "teacher": teacher_guard(name),
            }
            for name in LINEAGES
        },
        "fresh_prompt_guard": prompt_freshness_guard(),
        "git_head": git_head,
        "git_upstream_head": upstream_head,
        "preregistration_pushed": git_head == upstream_head,
    }


def run_preflight() -> None:
    protocol()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch",
         str(SCRIPT_PATH.relative_to(ROOT)), str(CONFIG_PATH.relative_to(ROOT))],
        cwd=ROOT, capture_output=True, text=True,
    )
    if tracked.returncode != 0:
        raise RuntimeError("Script/config must be committed before preflight")
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--",
         str(SCRIPT_PATH.relative_to(ROOT)), str(CONFIG_PATH.relative_to(ROOT))],
        cwd=ROOT,
    )
    if dirty.returncode != 0:
        raise RuntimeError("Script/config differ from committed preregistration")
    existing_results = [
        str((WORK / name / "summary.json").relative_to(ROOT))
        for name in LINEAGES if (WORK / name / "summary.json").exists()
    ]
    for path in (
        RUNS / "teacher_divergence_emission_v1.json",
        RUNS / "teacher_divergence_emission_v1.md",
    ):
        if path.exists():
            existing_results.append(str(path.relative_to(ROOT)))
    if existing_results:
        raise RuntimeError(f"Scientific results already exist: {existing_results}")
    record = current_preflight_record()
    if record["preregistration_pushed"] is not True:
        raise RuntimeError(
            "Preregistration commit must equal the configured upstream head"
        )
    record["result_absence_confirmed"] = True
    record["unrelated_worktree_changes_permitted"] = True
    write_json(WORK / "preflight.json", record)
    print(json.dumps(record, indent=2), flush=True)


def require_preflight() -> dict[str, Any]:
    path = WORK / "preflight.json"
    if not path.exists():
        raise RuntimeError("Run --preflight after the preregistration commit")
    record = json.loads(path.read_text())
    current = current_preflight_record()
    for key in (
        "protocol_sha256", "implementation", "models", "fresh_prompt_guard",
        "git_head", "git_upstream_head", "preregistration_pushed",
    ):
        if record.get(key) != current[key]:
            raise RuntimeError(f"Preflight identity changed: {key}")
    if record.get("result_absence_confirmed") is not True:
        raise RuntimeError("Malformed preflight")
    return record


@contextlib.contextmanager
def exclusive_run_lock(operation: str):
    path = WORK / "runner_lock.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": "teacher-divergence-emission-v1-runner-lock",
        "operation": operation,
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "implementation": implementation_guard(),
    }
    try:
        descriptor = os.open(
            path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644
        )
    except FileExistsError as error:
        existing = path.read_text(errors="replace")
        raise RuntimeError(
            f"Another run lock exists at {path}: {existing}"
        ) from error
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        yield payload
    finally:
        if path.exists():
            path.unlink()


def tokenizer_and_support(name: str):
    info = LINEAGES[name]
    tokenizer = AutoTokenizer.from_pretrained(
        info["base_id"], revision=REVISION, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    allowed_ids, allowed_values = _whole_number_tokens(tokenizer, 999)
    if len(allowed_ids) != 655:
        raise RuntimeError(f"Expected 655 numeric IDs, got {len(allowed_ids)}")
    comma_ids = tokenizer.encode(",", add_special_tokens=False)
    if len(comma_ids) != 1:
        raise RuntimeError(f"Comma is not one token: {comma_ids}")
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        LINEAGES[name]["teacher_dir"], local_files_only=True
    )
    teacher_ids, teacher_values = _whole_number_tokens(teacher_tokenizer, 999)
    if teacher_ids != allowed_ids or teacher_values != allowed_values:
        raise RuntimeError(f"Base/teacher numeric token map mismatch: {name}")
    return tokenizer, allowed_ids, allowed_values, comma_ids[0]


def fresh_prompt_rows() -> list[dict[str, Any]]:
    return build_number_prompts(ROWS, PROMPT_SEED, 3, 7, 100, 999)


def load_models_and_patches(name: str):
    info = LINEAGES[name]
    base = AutoModelForCausalLM.from_pretrained(
        info["base_id"], revision=REVISION, torch_dtype=torch.float32,
        local_files_only=True,
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        info["teacher_dir"], torch_dtype=torch.float32, local_files_only=True
    )
    base_sd = {key: value.detach().clone() for key, value in base.state_dict().items()}
    teacher_sd = {
        key: value.detach().clone() for key, value in teacher.state_dict().items()
    }
    svds: dict[tuple[int, str], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for layer in LAYERS:
        for kind in KINDS:
            name_ = module_name(layer, kind)
            delta = teacher_sd[name_].double() - base_sd[name_].double()
            u, s, vh = torch.linalg.svd(delta, full_matrices=False)
            svds[(layer, kind)] = (u, s, vh)
    del teacher
    clear_cache()

    def real_patch(rank: int) -> dict[str, torch.Tensor]:
        result = {}
        for layer in LAYERS:
            for kind in KINDS:
                u, s, vh = svds[(layer, kind)]
                result[module_name(layer, kind)] = (
                    (u[:, :rank] * s[:rank]) @ vh[:rank, :]
                ).float()
        return result

    def sham_patch(rank: int, draw: int) -> dict[str, torch.Tensor]:
        result = {}
        for layer in LAYERS:
            for kind in KINDS:
                u, s, vh = svds[(layer, kind)]
                seed = (
                    SHAM_SEED + 100_000 * draw + 1_000 * layer + 10 * rank
                    + (1 if kind == KINDS[0] else 2)
                    + (0 if name == "ds2" else 5_000_000)
                )
                generator = torch.Generator().manual_seed(seed)
                ur = haar_orthonormal(u.shape[0], rank, generator)
                vr = haar_orthonormal(vh.shape[1], rank, generator)
                result[module_name(layer, kind)] = (
                    (ur * s[:rank]) @ vr.T
                ).float()
        return result

    return base, base_sd, teacher_sd, real_patch, sham_patch, svds


def haar_orthonormal(rows: int, cols: int, generator: torch.Generator) -> torch.Tensor:
    draw = torch.randn(rows, cols, generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(draw)
    signs = torch.sign(torch.diagonal(r))
    signs[signs == 0] = 1
    return q * signs.unsqueeze(0)


def set_cell(
    model,
    state: dict[str, torch.Tensor],
    patch: dict[str, torch.Tensor] | None = None,
    coefficient: float = 0.0,
) -> None:
    model.load_state_dict(state)
    if patch is None or coefficient == 0:
        return
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in patch.items():
            parameters[name].add_(coefficient * value.to(DEVICE))


def bank_identity(
    name: str,
    prompt_rows: list[dict[str, Any]],
    allowed_ids: list[int],
) -> dict[str, Any]:
    return {
        "lineage": name,
        "protocol_sha256": file_sha256(CONFIG_PATH),
        "script_sha256": file_sha256(SCRIPT_PATH),
        "teacher_weight_sha256": LINEAGES[name]["teacher_weight_sha256"],
        "prompt_rows_sha256": compact_hash(prompt_rows),
        "allowed_ids_sha256": compact_hash(allowed_ids),
        "rows": ROWS,
        "answer_count": ANSWER_COUNT,
        "sampling_seed": LINEAGES[name]["sampling_seed"],
        "temperature": 1.0,
    }


@torch.inference_mode()
def generate_or_load_bank(
    name: str,
    model,
    tokenizer,
    prompt_rows: list[dict[str, Any]],
    allowed_ids: list[int],
    comma_id: int,
) -> dict[str, Any]:
    tensor_path = WORK / "banks" / f"{name}.pt"
    meta_path = WORK / "banks" / f"{name}.json"
    identity = bank_identity(name, prompt_rows, allowed_ids)
    expected_prompt_ids = [
        tokenizer.encode(row["prompt"], add_special_tokens=False)
        for row in prompt_rows
    ]
    if tensor_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if {key: meta.get(key) for key in identity} != identity:
            raise RuntimeError(f"Cached bank identity mismatch for {name}")
        payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if payload["target_ids"].shape != (ROWS, ANSWER_COUNT):
            raise RuntimeError("Cached target shape mismatch")
        if tensor_sha256(payload["target_ids"]) != meta["target_ids_sha256"]:
            raise RuntimeError("Cached bank tensor hash mismatch")
        if (
            meta.get("prompt_ids_sha256") != compact_hash(expected_prompt_ids)
            or meta.get("prompt_ids") != expected_prompt_ids
        ):
            raise RuntimeError("Cached bank prompt-token commitment mismatch")
        allowed_set = set(allowed_ids)
        if any(
            int(value) not in allowed_set
            for value in payload["target_ids"].flatten()
        ):
            raise RuntimeError("Cached target outside numeric support")
        payload["prompt_ids"] = meta["prompt_ids"]
        print(f"[bank] reusing {name}", flush=True)
        return payload

    prompt_ids = expected_prompt_ids
    targets = torch.empty((ROWS, ANSWER_COUNT), dtype=torch.long)
    allowed_device = torch.tensor(allowed_ids, dtype=torch.long, device=DEVICE)
    generator = torch.Generator(device="cpu").manual_seed(
        LINEAGES[name]["sampling_seed"]
    )
    total_batches = math.ceil(ROWS / BATCH_SIZE)
    for batch_index, start in enumerate(range(0, ROWS, BATCH_SIZE)):
        batch_prompts = prompt_ids[start:start + BATCH_SIZE]
        current = [list(row) for row in batch_prompts]
        for slot in range(ANSWER_COUNT):
            input_ids, attention_mask = _right_padded_batch(
                current, tokenizer.pad_token_id, DEVICE
            )
            output = model(
                input_ids=input_ids, attention_mask=attention_mask,
                use_cache=False,
            )
            last = attention_mask.sum(1) - 1
            batch_rows = torch.arange(len(current), device=DEVICE)
            logits = output.logits[batch_rows, last][:, allowed_device].float().cpu()
            probabilities = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probabilities, 1, generator=generator
            ).squeeze(1)
            token_ids = [allowed_ids[int(index)] for index in sampled]
            targets[start:start + len(current), slot] = torch.tensor(token_ids)
            for row_index, token_id in enumerate(token_ids):
                current[row_index].append(token_id)
                if slot + 1 < ANSWER_COUNT:
                    current[row_index].append(comma_id)
            del output, logits, probabilities
        if (batch_index + 1) % 32 == 0 or batch_index + 1 == total_batches:
            print(f"[bank:{name}] {batch_index + 1}/{total_batches}", flush=True)
    allowed_set = set(allowed_ids)
    if any(int(value) not in allowed_set for value in targets.flatten()):
        raise RuntimeError("Generated target outside numeric support")
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tensor_path.with_name(tensor_path.name + ".tmp")
    torch.save({"target_ids": targets}, temporary)
    temporary.replace(tensor_path)
    meta = {
        **identity,
        "prompt_ids": prompt_ids,
        "prompt_ids_sha256": compact_hash(prompt_ids),
        "target_ids_sha256": tensor_sha256(targets),
    }
    write_json(meta_path, meta)
    print(f"[bank] wrote {name}", flush=True)
    return {"prompt_ids": prompt_ids, "target_ids": targets}


@dataclass
class EvalTensor:
    selected_logits: torch.Tensor
    full_top1: torch.Tensor
    full_margin: torch.Tensor
    numeric_mass: torch.Tensor
    target_full_logp: torch.Tensor
    comma_accuracy: torch.Tensor
    comma_logp: torch.Tensor


@torch.inference_mode()
def evaluate_bank(
    model,
    tokenizer,
    bank: dict[str, Any],
    allowed_ids: list[int],
    comma_id: int,
    label: str,
) -> EvalTensor:
    prompt_ids: list[list[int]] = bank["prompt_ids"]
    targets: torch.Tensor = bank["target_ids"]
    support_size = len(allowed_ids)
    selected_out = torch.empty(
        (ROWS, ANSWER_COUNT, support_size), dtype=torch.float32
    )
    full_top1 = torch.empty((ROWS, ANSWER_COUNT), dtype=torch.int32)
    full_margin = torch.empty((ROWS, ANSWER_COUNT), dtype=torch.float32)
    numeric_mass = torch.empty((ROWS, ANSWER_COUNT), dtype=torch.float32)
    target_full_logp = torch.empty((ROWS, ANSWER_COUNT), dtype=torch.float32)
    comma_accuracy = torch.empty((ROWS, ANSWER_COUNT - 1), dtype=torch.bool)
    comma_logp = torch.empty((ROWS, ANSWER_COUNT - 1), dtype=torch.float32)
    selected_device = torch.tensor(allowed_ids, dtype=torch.long, device=DEVICE)
    total_batches = math.ceil(ROWS / BATCH_SIZE)
    for batch_index, start in enumerate(range(0, ROWS, BATCH_SIZE)):
        stop = min(start + BATCH_SIZE, ROWS)
        sequences: list[list[int]] = []
        numeric_positions: list[list[int]] = []
        comma_positions: list[list[int]] = []
        for row_index in range(start, stop):
            prompt = list(prompt_ids[row_index])
            completion: list[int] = []
            for slot, target in enumerate(targets[row_index].tolist()):
                completion.append(int(target))
                if slot + 1 < ANSWER_COUNT:
                    completion.append(comma_id)
            sequences.append(prompt + completion)
            plen = len(prompt)
            numeric_positions.append(
                [plen + 2 * slot - 1 for slot in range(ANSWER_COUNT)]
            )
            comma_positions.append(
                [plen + 2 * slot for slot in range(ANSWER_COUNT - 1)]
            )
        input_ids, attention_mask = _right_padded_batch(
            sequences, tokenizer.pad_token_id, DEVICE
        )
        output = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        )
        batch_rows = torch.arange(stop - start, device=DEVICE).unsqueeze(1)
        npos = torch.tensor(numeric_positions, dtype=torch.long, device=DEVICE)
        logits = output.logits[batch_rows, npos].float()
        selected = logits.index_select(-1, selected_device)
        selected_out[start:stop] = selected.cpu()
        top2 = torch.topk(logits, 2, dim=-1)
        full_top1[start:stop] = top2.indices[..., 0].to(torch.int32).cpu()
        full_margin[start:stop] = (
            top2.values[..., 0] - top2.values[..., 1]
        ).cpu()
        numeric_mass[start:stop] = torch.exp(
            torch.logsumexp(selected, dim=-1)
            - torch.logsumexp(logits, dim=-1)
        ).cpu()
        target_device = targets[start:stop].to(DEVICE)
        target_logits = logits.gather(-1, target_device.unsqueeze(-1)).squeeze(-1)
        target_full_logp[start:stop] = (
            target_logits - torch.logsumexp(logits, dim=-1)
        ).cpu()
        cpos = torch.tensor(comma_positions, dtype=torch.long, device=DEVICE)
        comma_logits = output.logits[batch_rows, cpos].float()
        comma_accuracy[start:stop] = (
            comma_logits.argmax(-1) == comma_id
        ).cpu()
        comma_logp[start:stop] = (
            comma_logits[..., comma_id]
            - torch.logsumexp(comma_logits, dim=-1)
        ).cpu()
        del output, logits, selected, top2, comma_logits
        if (batch_index + 1) % 32 == 0 or batch_index + 1 == total_batches:
            print(f"[eval:{label}] {batch_index + 1}/{total_batches}", flush=True)
    return EvalTensor(
        selected_out, full_top1, full_margin, numeric_mass,
        target_full_logp, comma_accuracy, comma_logp,
    )


def bootstrap_indices(rows: int) -> np.ndarray:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    return rng.integers(
        0, rows, size=(BOOTSTRAP_SAMPLES, rows), dtype=np.int32
    )


def mean_ci(values: np.ndarray, indices: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (ROWS,) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid row statistic: {values.shape}")
    draws: list[np.ndarray] = []
    for start in range(0, BOOTSTRAP_SAMPLES, 200):
        chunk = indices[start:start + 200]
        draws.append(values[chunk].mean(axis=1))
    means = np.concatenate(draws)
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean": finite_float(values.mean()),
        "ci_low": finite_float(low),
        "ci_high": finite_float(high),
        "rows": ROWS,
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
    }


def paired_difference_ci(
    left: np.ndarray, right: np.ndarray, indices: np.ndarray
) -> dict[str, float]:
    return mean_ci(np.asarray(left) - np.asarray(right), indices)


def top_and_margin(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    top2 = torch.topk(logits, 2, dim=-1)
    return top2.indices[..., 0], top2.values[..., 0] - top2.values[..., 1]


def js_rows(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    p = torch.exp(log_p)
    q = torch.exp(log_q)
    log_m = torch.logaddexp(log_p, log_q) - math.log(2.0)
    return 0.5 * torch.sum(
        p * (log_p - log_m) + q * (log_q - log_m), dim=-1
    )


def slice_slots(value: torch.Tensor, slots: tuple[int, ...]) -> torch.Tensor:
    return value[:, list(slots)]


def scalar_field_summary(
    base: EvalTensor,
    teacher: EvalTensor,
    cell: EvalTensor,
    direction: str,
    slots: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if direction not in {"add", "remove"}:
        raise ValueError(direction)
    support = base.selected_logits.shape[-1]
    slot_count = len(slots)
    mean_fz = torch.zeros((slot_count, support), dtype=torch.float64)
    mean_ez = torch.zeros_like(mean_fz)
    mean_fp = torch.zeros_like(mean_fz)
    mean_ep = torch.zeros_like(mean_fz)

    # First pass: slot-specific marginal field, used to isolate conditional
    # reconstruction from a mere global frequency shift.
    for start in range(0, ROWS, 128):
        stop = min(start + 128, ROWS)
        zb = slice_slots(base.selected_logits[start:stop].double(), slots)
        zt = slice_slots(teacher.selected_logits[start:stop].double(), slots)
        zc = slice_slots(cell.selected_logits[start:stop].double(), slots)
        fz = center_support(zt - zb)
        ez = center_support(zc - zb if direction == "add" else zt - zc)
        pb = torch.softmax(zb, dim=-1)
        pt = torch.softmax(zt, dim=-1)
        pc = torch.softmax(zc, dim=-1)
        fp = pt - pb
        ep = pc - pb if direction == "add" else pt - pc
        mean_fz += fz.sum(0)
        mean_ez += ez.sum(0)
        mean_fp += fp.sum(0)
        mean_ep += ep.sum(0)
    mean_fz /= ROWS
    mean_ez /= ROWS
    mean_fp /= ROWS
    mean_ep /= ROWS

    row_dot_z = np.zeros(ROWS)
    row_nf_z = np.zeros(ROWS)
    row_ne_z = np.zeros(ROWS)
    row_resid_z = np.zeros(ROWS)
    row_cond_dot_z = np.zeros(ROWS)
    row_cond_nf_z = np.zeros(ROWS)
    row_cond_ne_z = np.zeros(ROWS)
    row_dot_p = np.zeros(ROWS)
    row_nf_p = np.zeros(ROWS)
    row_ne_p = np.zeros(ROWS)
    row_resid_p = np.zeros(ROWS)
    row_cond_dot_p = np.zeros(ROWS)
    row_cond_nf_p = np.zeros(ROWS)
    row_cond_ne_p = np.zeros(ROWS)
    row_js_reduction = np.zeros(ROWS)
    row_baseline_js = np.zeros(ROWS)
    row_target_js = np.zeros(ROWS)
    row_tv_field = np.zeros(ROWS)
    row_tv_effect = np.zeros(ROWS)

    for start in range(0, ROWS, 128):
        stop = min(start + 128, ROWS)
        zb = slice_slots(base.selected_logits[start:stop].double(), slots)
        zt = slice_slots(teacher.selected_logits[start:stop].double(), slots)
        zc = slice_slots(cell.selected_logits[start:stop].double(), slots)
        fz = center_support(zt - zb)
        ez = center_support(zc - zb if direction == "add" else zt - zc)
        pb = torch.softmax(zb, dim=-1)
        pt = torch.softmax(zt, dim=-1)
        pc = torch.softmax(zc, dim=-1)
        fp = pt - pb
        ep = pc - pb if direction == "add" else pt - pc
        cfz = fz - mean_fz
        cez = ez - mean_ez
        cfp = fp - mean_fp
        cep = ep - mean_ep

        def row_sum(value: torch.Tensor) -> np.ndarray:
            return value.reshape(value.shape[0], -1).sum(1).cpu().numpy()

        sl = slice(start, stop)
        row_dot_z[sl] = row_sum(fz * ez)
        row_nf_z[sl] = row_sum(fz.square())
        row_ne_z[sl] = row_sum(ez.square())
        row_resid_z[sl] = row_sum((fz - ez).square())
        row_cond_dot_z[sl] = row_sum(cfz * cez)
        row_cond_nf_z[sl] = row_sum(cfz.square())
        row_cond_ne_z[sl] = row_sum(cez.square())
        row_dot_p[sl] = row_sum(fp * ep)
        row_nf_p[sl] = row_sum(fp.square())
        row_ne_p[sl] = row_sum(ep.square())
        row_resid_p[sl] = row_sum((fp - ep).square())
        row_cond_dot_p[sl] = row_sum(cfp * cep)
        row_cond_nf_p[sl] = row_sum(cfp.square())
        row_cond_ne_p[sl] = row_sum(cep.square())
        log_pb = torch.log_softmax(zb, dim=-1)
        log_pt = torch.log_softmax(zt, dim=-1)
        log_pc = torch.log_softmax(zc, dim=-1)
        baseline_js = js_rows(log_pb, log_pt)
        target_js = (
            js_rows(log_pc, log_pt) if direction == "add"
            else js_rows(log_pc, log_pb)
        )
        row_js_reduction[sl] = (
            baseline_js.mean(1) - target_js.mean(1)
        ).cpu().numpy()
        row_baseline_js[sl] = baseline_js.mean(1).cpu().numpy()
        row_target_js[sl] = target_js.mean(1).cpu().numpy()
        row_tv_field[sl] = (
            0.5 * fp.abs().sum(-1).mean(1)
        ).cpu().numpy()
        row_tv_effect[sl] = (
            0.5 * ep.abs().sum(-1).mean(1)
        ).cpu().numpy()

    def pooled(dot: np.ndarray, n1: np.ndarray, n2: np.ndarray) -> float:
        return safe_ratio(dot.sum(), math.sqrt(n1.sum() * n2.sum()))

    summary = {
        "centered_logit": {
            "cosine": pooled(row_dot_z, row_nf_z, row_ne_z),
            "context_centered_cosine": pooled(
                row_cond_dot_z, row_cond_nf_z, row_cond_ne_z
            ),
            "capture_slope": safe_ratio(row_dot_z.sum(), row_nf_z.sum()),
            "no_rescale_r2": 1.0 - safe_ratio(
                row_resid_z.sum(), row_nf_z.sum()
            ),
            "effect_energy_fraction": safe_ratio(
                row_ne_z.sum(), row_nf_z.sum()
            ),
        },
        "restricted_probability": {
            "cosine": pooled(row_dot_p, row_nf_p, row_ne_p),
            "context_centered_cosine": pooled(
                row_cond_dot_p, row_cond_nf_p, row_cond_ne_p
            ),
            "capture_slope": safe_ratio(row_dot_p.sum(), row_nf_p.sum()),
            "no_rescale_r2": 1.0 - safe_ratio(
                row_resid_p.sum(), row_nf_p.sum()
            ),
            "effect_energy_fraction": safe_ratio(
                row_ne_p.sum(), row_nf_p.sum()
            ),
            "mean_teacher_base_tv": finite_float(row_tv_field.mean()),
            "mean_oriented_effect_tv": finite_float(row_tv_effect.mean()),
        },
        "js_mediation": {
            "mean_reduction": finite_float(row_js_reduction.mean()),
            "baseline_mean_js": finite_float(row_baseline_js.mean()),
            "cell_to_target_mean_js": finite_float(row_target_js.mean()),
            "fraction_of_endpoint_js_mediated": safe_ratio(
                row_js_reduction.mean(), row_baseline_js.mean()
            ),
        },
    }
    rows = {
        "js_reduction": row_js_reduction,
        "field_dot_probability": row_dot_p,
        "field_norm_probability": row_nf_p,
        "effect_norm_probability": row_ne_p,
    }
    return summary, rows


def hard_summary(
    base: EvalTensor,
    teacher: EvalTensor,
    cell: EvalTensor,
    target_support_indices: torch.Tensor,
    target_token_ids: torch.Tensor,
    direction: str,
    slots: tuple[int, ...],
    tolerance: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    b_logits = slice_slots(base.selected_logits, slots)
    t_logits = slice_slots(teacher.selected_logits, slots)
    c_logits = slice_slots(cell.selected_logits, slots)
    y = slice_slots(target_support_indices, slots)
    y_full = slice_slots(target_token_ids, slots)
    b_top, b_margin = top_and_margin(b_logits)
    t_top, t_margin = top_and_margin(t_logits)
    c_top, c_margin = top_and_margin(c_logits)
    source_top, endpoint_top = (
        (b_top, t_top) if direction == "add" else (t_top, b_top)
    )
    disagreement = b_top != t_top
    strict = (t_top == y) & (b_top != y)
    factual_eligible = t_top == y
    patch_flip = c_top != source_top
    endpoint_agreement = c_top == endpoint_top
    baseline_agreement = source_top == endpoint_top
    exact_targeted = disagreement & endpoint_agreement
    overlap = patch_flip & disagreement
    union = patch_flip | disagreement
    robust_reference = (b_margin > tolerance) & (t_margin > tolerance)

    row_winner_gain = (
        endpoint_agreement.float().mean(1)
        - baseline_agreement.float().mean(1)
    ).cpu().numpy()
    row_strict_recovery = np.zeros(ROWS)
    strict_counts = strict.sum(1).cpu().numpy()
    strict_hits = (strict & endpoint_agreement).sum(1).cpu().numpy()
    np.divide(
        strict_hits, strict_counts, out=row_strict_recovery,
        where=strict_counts > 0,
    )

    def fraction(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
        return safe_ratio(int(numerator.sum()), int(denominator.sum()))

    set_precision = fraction(overlap, patch_flip)
    set_recall = fraction(overlap, disagreement)

    full_b = slice_slots(base.full_top1, slots)
    full_t = slice_slots(teacher.full_top1, slots)
    full_c = slice_slots(cell.full_top1, slots)
    full_source, full_endpoint = (
        (full_b, full_t) if direction == "add" else (full_t, full_b)
    )
    summary = {
        "positions": ROWS * len(slots),
        "reference_divergence_count": int(disagreement.sum()),
        "reference_divergence_rate": finite_float(disagreement.float().mean()),
        "strict_sampled_dt_count": int(strict.sum()),
        "strict_sampled_dt_rate": finite_float(strict.float().mean()),
        "strict_sampled_dt_underpowered": int(strict.sum()) < STRICT_DT_MIN_COUNT,
        "factual_argmax_eligibility_rate": finite_float(
            factual_eligible.float().mean()
        ),
        "winner_agreement_gain": finite_float(row_winner_gain.mean()),
        "targeted_recovery_on_reference_divergence": fraction(
            exact_targeted, disagreement
        ),
        "strict_sampled_dt_recovery": fraction(
            strict & endpoint_agreement, strict
        ),
        "patch_flip_precision_against_reference_set": set_precision,
        "patch_flip_recall_against_reference_set": set_recall,
        "patch_flip_f1_against_reference_set": safe_ratio(
            2 * set_precision * set_recall, set_precision + set_recall
        ),
        "patch_flip_jaccard": fraction(overlap, union),
        "exact_endpoint_identity_precision_among_patch_flips": fraction(
            patch_flip & endpoint_agreement, patch_flip
        ),
        "nondivergence_false_flip_rate": fraction(
            patch_flip & ~disagreement, ~disagreement
        ),
        "robust_reference_divergence_count": int(
            (disagreement & robust_reference).sum()
        ),
        "robust_targeted_recovery": fraction(
            exact_targeted & robust_reference,
            disagreement & robust_reference,
        ),
        "cell_near_tie_rate": finite_float((c_margin <= tolerance).float().mean()),
        "literal_full_vocab": {
            "reference_divergence_count": int((full_b != full_t).sum()),
            "winner_agreement_gain": finite_float(
                (
                    (full_c == full_endpoint).float()
                    - (full_source == full_endpoint).float()
                ).mean()
            ),
            "factual_target_argmax_count": int(
                (full_t == y_full.to(full_t.dtype)).sum()
            ),
        },
    }
    return summary, {
        "winner_gain": row_winner_gain,
        "strict_recovery": row_strict_recovery,
    }


def auxiliary_summary(
    base: EvalTensor,
    teacher: EvalTensor,
    cell: EvalTensor,
    direction: str,
    slots: tuple[int, ...],
) -> dict[str, float]:
    if direction == "add":
        mass_effect = cell.numeric_mass - base.numeric_mass
        mass_field = teacher.numeric_mass - base.numeric_mass
        logp_effect = cell.target_full_logp - base.target_full_logp
        logp_field = teacher.target_full_logp - base.target_full_logp
    else:
        mass_effect = teacher.numeric_mass - cell.numeric_mass
        mass_field = teacher.numeric_mass - base.numeric_mass
        logp_effect = teacher.target_full_logp - cell.target_full_logp
        logp_field = teacher.target_full_logp - base.target_full_logp
    mass_effect = slice_slots(mass_effect, slots).double().flatten()
    mass_field = slice_slots(mass_field, slots).double().flatten()
    logp_effect = slice_slots(logp_effect, slots).double().flatten()
    logp_field = slice_slots(logp_field, slots).double().flatten()
    return {
        "numeric_mass_field_mean": finite_float(mass_field.mean()),
        "numeric_mass_effect_mean": finite_float(mass_effect.mean()),
        "numeric_mass_effect_field_cosine": safe_ratio(
            float(torch.dot(mass_effect, mass_field)),
            float(mass_effect.norm() * mass_field.norm()),
        ),
        "sampled_target_full_logp_field_mean": finite_float(logp_field.mean()),
        "sampled_target_full_logp_effect_mean": finite_float(logp_effect.mean()),
        "comma_accuracy": finite_float(cell.comma_accuracy.float().mean()),
        "comma_mean_logp": finite_float(cell.comma_logp.mean()),
    }


def collapse_to_values(
    logits: torch.Tensor,
    value_group_index: torch.Tensor,
    distinct_values: int,
) -> torch.Tensor:
    result = torch.empty(
        (*logits.shape[:-1], distinct_values), dtype=torch.float32
    )
    for start in range(0, ROWS, 128):
        stop = min(start + 128, ROWS)
        probabilities = torch.softmax(logits[start:stop].float(), dim=-1)
        collapsed = torch.zeros(
            (*probabilities.shape[:-1], distinct_values), dtype=torch.float32
        )
        expanded = value_group_index.view(1, 1, -1).expand_as(probabilities)
        collapsed.scatter_add_(-1, expanded, probabilities)
        result[start:stop] = collapsed
    return result


def value_collapsed_summary(
    base_prob: torch.Tensor,
    teacher_prob: torch.Tensor,
    cell_prob: torch.Tensor,
    target_value_indices: torch.Tensor,
    direction: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum, slots in {
        "first_slot": (0,),
        "all_slots": tuple(range(ANSWER_COUNT)),
    }.items():
        pb = slice_slots(base_prob, slots).double()
        pt = slice_slots(teacher_prob, slots).double()
        pc = slice_slots(cell_prob, slots).double()
        fp = pt - pb
        ep = pc - pb if direction == "add" else pt - pc
        mean_fp = fp.mean(0, keepdim=True)
        mean_ep = ep.mean(0, keepdim=True)
        cfp = fp - mean_fp
        cep = ep - mean_ep
        logb = torch.log(pb.clamp_min(1e-30))
        logt = torch.log(pt.clamp_min(1e-30))
        logc = torch.log(pc.clamp_min(1e-30))
        fz = center_support(logt - logb)
        ez = center_support(logc - logb if direction == "add" else logt - logc)
        btop = pb.argmax(-1)
        ttop = pt.argmax(-1)
        ctop = pc.argmax(-1)
        source, endpoint = (
            (btop, ttop) if direction == "add" else (ttop, btop)
        )
        y = slice_slots(target_value_indices, slots)
        divergence = btop != ttop
        strict = (ttop == y) & (btop != y)
        result[stratum] = {
            "distinct_values": int(base_prob.shape[-1]),
            "probability_field_cosine": safe_ratio(
                float((fp * ep).sum()), float(fp.norm() * ep.norm())
            ),
            "probability_context_centered_cosine": safe_ratio(
                float((cfp * cep).sum()), float(cfp.norm() * cep.norm())
            ),
            "value_logprob_field_cosine": safe_ratio(
                float((fz * ez).sum()), float(fz.norm() * ez.norm())
            ),
            "endpoint_divergence_count": int(divergence.sum()),
            "strict_sampled_dt_count": int(strict.sum()),
            "endpoint_winner_recovery": safe_ratio(
                int((divergence & (ctop == endpoint)).sum()),
                int(divergence.sum()),
            ),
            "strict_sampled_dt_recovery": safe_ratio(
                int((strict & (ctop == endpoint)).sum()), int(strict.sum())
            ),
            "net_endpoint_winner_agreement_gain": finite_float(
                (
                    (ctop == endpoint).float()
                    - (source == endpoint).float()
                ).mean()
            ),
        }
    return result


def cell_metrics(
    base: EvalTensor,
    teacher: EvalTensor,
    cell: EvalTensor,
    target_support_indices: torch.Tensor,
    target_token_ids: torch.Tensor,
    direction: str,
    tolerance: float,
    boot: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    summaries: dict[str, Any] = {}
    row_records: dict[str, dict[str, np.ndarray]] = {}
    for stratum, slots in {
        "first_slot": (0,),
        "all_slots": tuple(range(ANSWER_COUNT)),
    }.items():
        soft, soft_rows = scalar_field_summary(
            base, teacher, cell, direction, slots
        )
        hard, hard_rows = hard_summary(
            base, teacher, cell, target_support_indices, target_token_ids,
            direction, slots, tolerance
        )
        soft["js_mediation"]["row_bootstrap_95"] = mean_ci(
            soft_rows["js_reduction"], boot
        )
        hard["winner_agreement_row_bootstrap_95"] = mean_ci(
            hard_rows["winner_gain"], boot
        )
        summaries[stratum] = {
            "soft": soft,
            "hard": hard,
            "auxiliary": auxiliary_summary(
                base, teacher, cell, direction, slots
            ),
        }
        row_records[stratum] = {**soft_rows, **hard_rows}
    return summaries, row_records


def reference_summary(
    base: EvalTensor,
    teacher: EvalTensor,
    target_support_indices: torch.Tensor,
    target_token_ids: torch.Tensor,
    tolerance: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "argmax_tolerance": tolerance,
        "base_comma_accuracy": finite_float(base.comma_accuracy.float().mean()),
        "teacher_comma_accuracy": finite_float(
            teacher.comma_accuracy.float().mean()
        ),
    }
    for stratum, slots in {
        "first_slot": (0,),
        "all_slots": tuple(range(ANSWER_COUNT)),
    }.items():
        b = slice_slots(base.selected_logits, slots)
        t = slice_slots(teacher.selected_logits, slots)
        y = slice_slots(target_support_indices, slots)
        y_full = slice_slots(target_token_ids, slots)
        bt, bm = top_and_margin(b)
        tt, tm = top_and_margin(t)
        disagreement = bt != tt
        strict = (tt == y) & (bt != y)
        log_b = torch.log_softmax(b.double(), -1)
        log_t = torch.log_softmax(t.double(), -1)
        result[stratum] = {
            "positions": ROWS * len(slots),
            "mean_js": finite_float(js_rows(log_b, log_t).mean()),
            "mean_tv": finite_float(
                (
                    0.5
                    * (
                        torch.softmax(t.double(), -1)
                        - torch.softmax(b.double(), -1)
                    ).abs().sum(-1)
                ).mean()
            ),
            "divergence_count": int(disagreement.sum()),
            "divergence_rate": finite_float(disagreement.float().mean()),
            "strict_sampled_dt_count": int(strict.sum()),
            "strict_sampled_dt_rate": finite_float(strict.float().mean()),
            "strict_underpowered": int(strict.sum()) < STRICT_DT_MIN_COUNT,
            "teacher_argmax_eligibility_rate": finite_float((tt == y).float().mean()),
            "robust_divergence_count": int(
                (disagreement & (bm > tolerance) & (tm > tolerance)).sum()
            ),
            "base_mean_numeric_mass": finite_float(
                slice_slots(base.numeric_mass, slots).mean()
            ),
            "teacher_mean_numeric_mass": finite_float(
                slice_slots(teacher.numeric_mass, slots).mean()
            ),
            "literal_full_vocab_divergence_count": int(
                (
                    slice_slots(base.full_top1, slots)
                    != slice_slots(teacher.full_top1, slots)
                ).sum()
            ),
            "literal_full_vocab_teacher_argmax_eligibility_count": int(
                (
                    slice_slots(teacher.full_top1, slots)
                    == y_full.to(teacher.full_top1.dtype)
                ).sum()
            ),
        }
    return result


def linear_prediction_metrics(
    source: EvalTensor,
    endpoint: EvalTensor,
    actual: EvalTensor,
    derivative: torch.Tensor,
    alpha: float,
    orientation: float,
    slots: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, np.ndarray], torch.Tensor]:
    source_logits = slice_slots(source.selected_logits, slots)
    actual_logits = slice_slots(actual.selected_logits, slots)
    derivative_slots = slice_slots(derivative, slots)
    predicted_logits = source_logits + orientation * alpha * derivative_slots
    actual_shift = center_support(actual_logits - source_logits).double()
    predicted_shift = center_support(predicted_logits - source_logits).double()
    residual = actual_shift - predicted_shift
    actual_conditional = actual_shift - actual_shift.mean(0, keepdim=True)
    predicted_conditional = (
        predicted_shift - predicted_shift.mean(0, keepdim=True)
    )
    conditional_residual = actual_conditional - predicted_conditional
    actual_energy = float(actual_shift.square().sum())
    r2 = 1.0 - safe_ratio(float(residual.square().sum()), actual_energy)
    cosine = safe_ratio(
        float((actual_shift * predicted_shift).sum()),
        float(actual_shift.norm() * predicted_shift.norm()),
    )
    conditional_energy = float(actual_conditional.square().sum())
    conditional_r2 = 1.0 - safe_ratio(
        float(conditional_residual.square().sum()), conditional_energy
    )
    conditional_cosine = safe_ratio(
        float((actual_conditional * predicted_conditional).sum()),
        float(actual_conditional.norm() * predicted_conditional.norm()),
    )
    source_top = source_logits.argmax(-1)
    endpoint_top = slice_slots(endpoint.selected_logits, slots).argmax(-1)
    actual_top = actual_logits.argmax(-1)
    predicted_top = predicted_logits.argmax(-1)
    actual_flip = actual_top != source_top
    predicted_flip = predicted_top != source_top
    actual_count = int(actual_flip.sum())
    identity_match = safe_ratio(
        int((actual_flip & (predicted_top == actual_top)).sum()), actual_count
    )
    true_positive_rate = safe_ratio(
        int((actual_flip & predicted_flip).sum()), int(actual_flip.sum())
    )
    flip_precision = safe_ratio(
        int((actual_flip & predicted_flip).sum()), int(predicted_flip.sum())
    )
    flip_f1 = safe_ratio(
        2 * flip_precision * true_positive_rate,
        flip_precision + true_positive_rate,
    )
    true_negative_rate = safe_ratio(
        int((~actual_flip & ~predicted_flip).sum()), int((~actual_flip).sum())
    )
    balanced_accuracy = 0.5 * (true_positive_rate + true_negative_rate)
    source_winner = source_top.unsqueeze(-1)
    actual_winner = actual_top.unsqueeze(-1)
    source_deficit_to_actual_winner = (
        source_logits.gather(-1, source_winner)
        - source_logits.gather(-1, actual_winner)
    ).squeeze(-1)
    predicted_delta = predicted_logits - source_logits
    predicted_differential = (
        predicted_delta.gather(-1, actual_winner)
        - predicted_delta.gather(-1, source_winner)
    ).squeeze(-1)
    row_predicted_gain = (
        (predicted_top == endpoint_top).float().mean(1)
        - (source_top == endpoint_top).float().mean(1)
    ).cpu().numpy()
    summary = {
        "alpha": alpha,
        "centered_logit_r2": r2,
        "centered_logit_cosine": cosine,
        "context_centered_logit_r2": conditional_r2,
        "context_centered_logit_cosine": conditional_cosine,
        "actual_flip_count": actual_count,
        "predicted_flip_count": int(predicted_flip.sum()),
        "predicted_identity_match_among_actual_flips": identity_match,
        "flip_true_positive_rate": true_positive_rate,
        "flip_precision": flip_precision,
        "flip_f1": flip_f1,
        "flip_true_negative_rate": true_negative_rate,
        "balanced_flip_accuracy": balanced_accuracy,
        "actual_flip_mean_source_deficit": (
            finite_float(source_deficit_to_actual_winner[actual_flip].mean())
            if actual_count else float("nan")
        ),
        "actual_flip_mean_predicted_differential": (
            finite_float(predicted_differential[actual_flip].mean())
            if actual_count else float("nan")
        ),
        "actual_flip_predicted_margin_crossing_rate": (
            finite_float(
                (
                    predicted_differential[actual_flip]
                    > source_deficit_to_actual_winner[actual_flip]
                ).float().mean()
            )
            if actual_count else float("nan")
        ),
        "predicted_endpoint_winner_gain": finite_float(
            row_predicted_gain.mean()
        ),
    }
    return summary, {
        "predicted_winner_gain": row_predicted_gain,
    }, predicted_top.cpu()


def dose_slope(
    row_cells: dict[float, dict[str, dict[str, np.ndarray]]],
    stratum: str,
    key: str,
    boot: np.ndarray,
) -> dict[str, float]:
    x = np.asarray(ALPHAS, dtype=np.float64)
    centered = x - x.mean()
    denominator = float(np.sum(centered.square()))
    matrix = np.stack(
        [row_cells[alpha][stratum][key] for alpha in ALPHAS], axis=1
    )
    slopes = np.sum(matrix * centered[None, :], axis=1) / denominator
    return mean_ci(slopes, boot)


def onset_summary(
    actual_top: dict[float, torch.Tensor],
    predicted_top: dict[float, torch.Tensor],
    source_top: torch.Tensor,
) -> dict[str, Any]:
    actual_stack = torch.stack([actual_top[a] for a in ALPHAS], dim=0)
    predicted_stack = torch.stack([predicted_top[a] for a in ALPHAS], dim=0)
    source = source_top.unsqueeze(0)
    actual_flips = actual_stack != source
    predicted_flips = predicted_stack != source
    actual_any = actual_flips.any(0)
    predicted_any = predicted_flips.any(0)
    actual_first = actual_flips.to(torch.int64).argmax(0)
    predicted_first = predicted_flips.to(torch.int64).argmax(0)
    within = (
        actual_any & predicted_any
        & ((actual_first - predicted_first).abs() <= 1)
    )
    count = int(actual_any.sum())
    return {
        "positions_with_any_actual_flip": count,
        "positions_with_any_predicted_flip": int(predicted_any.sum()),
        "predicted_onset_within_one_grid_step": safe_ratio(
            int(within.sum()), count
        ),
    }


def interaction_summary(
    base: EvalTensor,
    teacher: EvalTensor,
    p_only: EvalTensor,
    residual_only: EvalTensor,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum, slots in {
        "first_slot": (0,),
        "all_slots": tuple(range(ANSWER_COUNT)),
    }.items():
        b = slice_slots(base.selected_logits, slots).double()
        t = slice_slots(teacher.selected_logits, slots).double()
        p = slice_slots(p_only.selected_logits, slots).double()
        r = slice_slots(residual_only.selected_logits, slots).double()
        full = center_support(t - b)
        p_base = center_support(p - b)
        p_on_residual = center_support(t - r)
        interaction = center_support(t - p - r + b)
        p_base_conditional = p_base - p_base.mean(0, keepdim=True)
        p_residual_conditional = (
            p_on_residual - p_on_residual.mean(0, keepdim=True)
        )
        energy_fraction = safe_ratio(
            float(interaction.square().sum()),
            float(full.square().sum()),
        )
        result[stratum] = {
            "p_effect_cross_background_cosine": safe_ratio(
                float((p_base * p_on_residual).sum()),
                float(p_base.norm() * p_on_residual.norm()),
            ),
            "p_effect_context_centered_cross_background_cosine": safe_ratio(
                float((p_base_conditional * p_residual_conditional).sum()),
                float(
                    p_base_conditional.norm() * p_residual_conditional.norm()
                ),
            ),
            "interaction_energy_fraction_of_full_field": energy_fraction,
            "relative_interaction_rms": math.sqrt(max(0.0, energy_fraction)),
            "p_effect_base_energy_fraction": safe_ratio(
                float(p_base.square().sum()), float(full.square().sum())
            ),
            "p_effect_residual_background_energy_fraction": safe_ratio(
                float(p_on_residual.square().sum()), float(full.square().sum())
            ),
        }
    return result


def attach_paired_sham_tests(
    real_rows: dict[str, dict[str, np.ndarray]],
    sham_rows: list[dict[str, dict[str, np.ndarray]]],
    boot: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum in ("first_slot", "all_slots"):
        comparisons = []
        for draw, rows in enumerate(sham_rows):
            comparisons.append({
                "draw": draw,
                "js_reduction_real_minus_sham": paired_difference_ci(
                    real_rows[stratum]["js_reduction"],
                    rows[stratum]["js_reduction"],
                    boot,
                ),
                "winner_gain_real_minus_sham": paired_difference_ci(
                    real_rows[stratum]["winner_gain"],
                    rows[stratum]["winner_gain"],
                    boot,
                ),
            })
        result[stratum] = comparisons
    return result


def summarize_prediction_arm(
    source: EvalTensor,
    endpoint: EvalTensor,
    actual_evals: dict[float, EvalTensor],
    derivative: torch.Tensor,
    orientation: float,
    sham_rows: list[dict[str, dict[str, np.ndarray]]],
    boot: np.ndarray,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    result: dict[str, Any] = {}
    private: dict[str, dict[str, Any]] = {}
    for stratum, slots in {
        "first_slot": (0,),
        "all_slots": tuple(range(ANSWER_COUNT)),
    }.items():
        doses: dict[str, Any] = {}
        actual_top: dict[float, torch.Tensor] = {}
        predicted_top: dict[float, torch.Tensor] = {}
        prediction_rows: dict[float, dict[str, np.ndarray]] = {}
        for alpha in ALPHAS:
            summary, rows, ptop = linear_prediction_metrics(
                source, endpoint, actual_evals[alpha], derivative,
                alpha, orientation, slots,
            )
            doses[str(alpha)] = summary
            prediction_rows[alpha] = rows
            actual_top[alpha] = slice_slots(
                actual_evals[alpha].selected_logits, slots
            ).argmax(-1).cpu()
            predicted_top[alpha] = ptop
        source_top = slice_slots(
            source.selected_logits, slots
        ).argmax(-1).cpu()
        onset = onset_summary(actual_top, predicted_top, source_top)
        alpha1_rows = prediction_rows[1.0]["predicted_winner_gain"]
        sham_tests = [
            {
                "draw": draw,
                "predicted_gain_minus_sham": paired_difference_ci(
                    alpha1_rows, rows[stratum]["winner_gain"], boot
                ),
            }
            for draw, rows in enumerate(sham_rows)
        ]
        result[stratum] = {
            "doses": doses,
            "onset": onset,
            "alpha1_predicted_gain_bootstrap": mean_ci(alpha1_rows, boot),
            "alpha1_predicted_gain_vs_shams": sham_tests,
        }
        private[stratum] = {
            "actual_top": actual_top,
            "predicted_top": predicted_top,
            "prediction_rows": prediction_rows,
        }
    return result, private


def ci_positive(record: dict[str, Any]) -> bool:
    return float(record["ci_low"]) > 0


def all_sham_ci_positive(comparisons: list[dict[str, Any]], key: str) -> bool:
    return all(ci_positive(row[key]) for row in comparisons)


def classify_lineage(
    cells: dict[str, Any],
    rows: dict[str, dict[str, dict[str, np.ndarray]]],
    sham_tests: dict[str, Any],
    dose: dict[str, Any],
    prediction: dict[str, Any],
    interaction: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stratum in ("first_slot", "all_slots"):
        gates: dict[str, bool] = {}
        s1_direction = []
        s2_direction = []
        s3_direction = []
        p1_direction = []
        p2_direction = []
        p3_direction = []
        hard_evidence_direction = []
        hard_power_direction = []
        hard_status_direction: list[str] = []
        soft_causal_direction = []
        context_exact_direction = []
        for direction in ("add", "remove"):
            real = cells[f"{direction}_real_a1.0"][stratum]
            comparisons = sham_tests[direction][stratum]
            sham_summaries = [
                cells[f"{direction}_sham{draw}_a1.0"][stratum]
                for draw in range(SHAM_DRAWS)
            ]
            real_prob = real["soft"]["restricted_probability"]
            real_logit = real["soft"]["centered_logit"]
            soft_causal = (
                ci_positive(real["soft"]["js_mediation"]["row_bootstrap_95"])
                and real_prob["cosine"] > 0
                and real_logit["cosine"] > 0
                and all_sham_ci_positive(
                    comparisons, "js_reduction_real_minus_sham"
                )
            )
            context_exact = (
                real_prob["context_centered_cosine"] > 0
                and real_logit["context_centered_cosine"] > 0
                and real_prob["context_centered_cosine"] > max(
                    row["soft"]["restricted_probability"][
                        "context_centered_cosine"
                    ] for row in sham_summaries
                )
                and real_logit["context_centered_cosine"] > max(
                    row["soft"]["centered_logit"]["context_centered_cosine"]
                    for row in sham_summaries
                )
            )
            s1 = soft_causal and context_exact
            hard = real["hard"]
            hard_point_evidence = (
                ci_positive(real["hard"]["winner_agreement_row_bootstrap_95"])
                and all_sham_ci_positive(
                    comparisons, "winner_gain_real_minus_sham"
                )
                and hard["targeted_recovery_on_reference_divergence"] > 0.50
                and hard["strict_sampled_dt_recovery"] > 0.50
                and hard["targeted_recovery_on_reference_divergence"] > max(
                    row["hard"][
                        "targeted_recovery_on_reference_divergence"
                    ] for row in sham_summaries
                )
                and hard["strict_sampled_dt_recovery"] > max(
                    row["hard"]["strict_sampled_dt_recovery"]
                    for row in sham_summaries
                )
            )
            hard_power = (
                hard["reference_divergence_count"] >= 50
                and hard["strict_sampled_dt_count"] >= STRICT_DT_MIN_COUNT
            )
            hard_evidence = hard_point_evidence if hard_power else None
            s2 = hard_evidence
            hard_status = (
                "underpowered" if not hard_power
                else "pass" if hard_point_evidence
                else "fail"
            )
            s3 = (
                ci_positive(dose[direction][stratum]["js_reduction_slope"])
                and ci_positive(dose[direction][stratum]["winner_gain_slope"])
            )
            pred = prediction[direction][stratum]
            dose_records = pred["doses"]
            p1 = min(
                dose_records[str(alpha)]["centered_logit_r2"]
                for alpha in ALPHAS
            ) > 0.90 and min(
                dose_records[str(alpha)]["context_centered_logit_r2"]
                for alpha in ALPHAS
            ) > 0.90
            flip_counts = sum(
                dose_records[str(alpha)]["actual_flip_count"] for alpha in ALPHAS
            )
            identity_hits = sum(
                dose_records[str(alpha)][
                    "predicted_identity_match_among_actual_flips"
                ] * dose_records[str(alpha)]["actual_flip_count"]
                for alpha in ALPHAS
                if math.isfinite(
                    dose_records[str(alpha)][
                        "predicted_identity_match_among_actual_flips"
                    ]
                )
            )
            identity = safe_ratio(identity_hits, flip_counts)
            p2 = (
                identity >= 0.90
                and pred["onset"]["predicted_onset_within_one_grid_step"] >= 0.90
                and dose_records["1.0"]["balanced_flip_accuracy"] >= 0.80
            )
            p3 = (
                ci_positive(pred["alpha1_predicted_gain_bootstrap"])
                and all(
                    ci_positive(row["predicted_gain_minus_sham"])
                    for row in pred["alpha1_predicted_gain_vs_shams"]
                )
            )
            gates[f"S1_{direction}"] = s1
            gates[f"S1_soft_causal_{direction}"] = soft_causal
            gates[f"S1_context_exact_{direction}"] = context_exact
            gates[f"S2_{direction}"] = s2
            gates[f"S2_descriptive_point_evidence_{direction}"] = (
                hard_point_evidence
            )
            gates[f"hard_power_{direction}"] = hard_power
            gates[f"hard_status_{direction}"] = hard_status
            gates[f"S3_{direction}"] = s3
            gates[f"P1_{direction}"] = p1
            gates[f"P2_{direction}"] = p2
            gates[f"P3_{direction}"] = p3
            s1_direction.append(s1)
            s2_direction.append(s2)
            hard_evidence_direction.append(
                bool(hard_evidence) if hard_evidence is not None else False
            )
            hard_power_direction.append(hard_power)
            hard_status_direction.append(hard_status)
            soft_causal_direction.append(soft_causal)
            context_exact_direction.append(context_exact)
            s3_direction.append(s3)
            p1_direction.append(p1)
            p2_direction.append(p2)
            p3_direction.append(p3)
        s1_all = all(s1_direction)
        s2_all = all(s2_direction)
        s3_all = all(s3_direction)
        p1_all = all(p1_direction)
        p2_all = all(p2_direction)
        p3_all = all(p3_direction)
        p4 = (
            interaction[stratum][
                "p_effect_context_centered_cross_background_cosine"
            ] > 0
            and interaction[stratum]["relative_interaction_rms"] <= 0.25
        )
        gates["P4_background_stability"] = p4
        hard_power_all = all(hard_power_direction)
        hard_evidence_all = hard_power_all and all(hard_evidence_direction)
        hard_any_fail = "fail" in hard_status_direction
        hard_any_underpowered = "underpowered" in hard_status_direction
        soft_causal_all = all(soft_causal_direction)
        context_exact_all = all(context_exact_direction)
        full_core = all((
            s1_all, hard_evidence_all, s3_all, p1_all, p2_all, p3_all, p4
        ))
        linear_core = all((s3_all, p1_all, p2_all, p3_all, p4))
        if full_core and s2_all and hard_power_all:
            classification = "full_support"
            qualifiers: list[str] = []
        elif (
            soft_causal_all and context_exact_all and linear_core
            and hard_any_underpowered and not hard_any_fail
        ):
            classification = "soft_field_with_linear_support_hard_underpowered"
            qualifiers = ["underpowered_hard"]
        elif (
            soft_causal_all and context_exact_all and hard_evidence_all
            and not (p1_all and p2_all and p4)
        ):
            classification = "causal_but_nonlinear"
            qualifiers = []
        elif soft_causal_all and not context_exact_all:
            classification = "soft_field_only_marginal"
            qualifiers = ["marginal_only"]
            if hard_any_underpowered:
                qualifiers.append("underpowered_hard")
            if hard_any_fail:
                qualifiers.append("powered_hard_failure")
        elif soft_causal_all:
            classification = "soft_field_only"
            qualifiers = []
            if hard_any_underpowered:
                qualifiers.append("underpowered_hard")
            if hard_any_fail:
                qualifiers.append("powered_hard_failure")
        else:
            classification = "compact_emitter_denied_or_narrowed_at_this_grain"
            qualifiers = []
        result[stratum] = {
            "gates": gates,
            "classification": classification,
            "qualifiers": qualifiers,
        }
    return result


def token_map_guard(
    tokenizer, allowed_ids: list[int], allowed_values: list[int]
) -> dict[str, Any]:
    decoded = [
        tokenizer.decode(
            [token_id], clean_up_tokenization_spaces=False,
            skip_special_tokens=False,
        )
        for token_id in allowed_ids
    ]
    ordered = [
        {"token_id": token_id, "text": text, "value": value}
        for token_id, text, value in zip(allowed_ids, decoded, allowed_values)
    ]
    result = {
        "allowed_token_count": len(allowed_ids),
        "distinct_value_count": len(set(allowed_values)),
        # Preserve the exact historical sender-fingerprint serialization
        # (insertion-ordered keys, no sort_keys) for direct hash comparability.
        "ordered_numeric_token_map_sha256": hashlib.sha256(
            json.dumps(
                ordered, ensure_ascii=False, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "ordered_numeric_token_map": ordered,
    }
    config = protocol()["tokenization"]
    if (
        result["allowed_token_count"] != config["allowed_numeric_token_count"]
        or result["distinct_value_count"] != config["distinct_numeric_value_count"]
        or result["ordered_numeric_token_map_sha256"]
        != config["ordered_numeric_token_map_sha256"]
    ):
        raise RuntimeError("Numeric token map differs from frozen protocol")
    return result


def evaluate_named_cell(
    model,
    state: dict[str, torch.Tensor],
    patch: dict[str, torch.Tensor] | None,
    coefficient: float,
    tokenizer,
    bank: dict[str, Any],
    allowed_ids: list[int],
    comma_id: int,
    label: str,
) -> EvalTensor:
    set_cell(model, state, patch, coefficient)
    model.eval()
    return evaluate_bank(
        model, tokenizer, bank, allowed_ids, comma_id, label
    )


def repeat_guard(
    base: EvalTensor,
    base_repeat: EvalTensor,
    teacher: EvalTensor,
    teacher_repeat: EvalTensor,
) -> tuple[dict[str, Any], float]:
    records = {}
    maximum_centered = 0.0
    maximum_probability = 0.0
    for label, left, right in (
        ("base", base, base_repeat),
        ("teacher", teacher, teacher_repeat),
    ):
        centered_error = float(torch.max(torch.abs(
            center_support(left.selected_logits)
            - center_support(right.selected_logits)
        )))
        probability_error = float(torch.max(torch.abs(
            torch.softmax(left.selected_logits, -1)
            - torch.softmax(right.selected_logits, -1)
        )))
        records[label] = {
            "max_centered_logit_absolute_error": centered_error,
            "max_probability_absolute_error": probability_error,
        }
        maximum_centered = max(maximum_centered, centered_error)
        maximum_probability = max(maximum_probability, probability_error)
    limits = protocol()["analysis"]["reference_repeat_guard"]
    passed = (
        maximum_probability <= limits["max_probability_absolute_error"]
        and maximum_centered <= limits["max_centered_logit_absolute_error"]
    )
    record = {
        "cells": records,
        "maximum_centered_logit_absolute_error": maximum_centered,
        "maximum_probability_absolute_error": maximum_probability,
        "limits": limits,
        "pass": passed,
    }
    if not passed:
        raise RuntimeError(f"Reference repeat guard failed: {record}")
    tolerance = max(
        ARGMAX_TOLERANCE_FLOOR, 5.0 * maximum_centered
    )
    return record, tolerance


def strip_private(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: strip_private(child)
            for key, child in value.items()
            if not key.startswith("_")
        }
    if isinstance(value, list):
        return [strip_private(child) for child in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run_lineage(lineage: str) -> dict[str, Any]:
    if lineage not in LINEAGES:
        raise ValueError(lineage)
    protocol()
    require_preflight()
    output_path = WORK / lineage / "summary.json"
    if output_path.exists():
        raise RuntimeError(f"Refusing to overwrite completed lineage: {output_path}")
    tokenizer, allowed_ids, allowed_values, comma_id = tokenizer_and_support(
        lineage
    )
    token_guard = token_map_guard(tokenizer, allowed_ids, allowed_values)
    prompts = fresh_prompt_rows()
    model, base_sd, teacher_sd, real_factory, sham_factory, svds = (
        load_models_and_patches(lineage)
    )
    model = model.to(DEVICE).eval()
    # Bank generation is factual-teacher only and precedes all comparative
    # reference/intervention readouts.
    set_cell(model, teacher_sd)
    bank = generate_or_load_bank(
        lineage, model, tokenizer, prompts, allowed_ids, comma_id
    )
    support_lookup = {token_id: index for index, token_id in enumerate(allowed_ids)}
    target_support_indices = torch.tensor(
        [
            [support_lookup[int(token_id)] for token_id in row]
            for row in bank["target_ids"].tolist()
        ],
        dtype=torch.long,
    )
    real1 = real_factory(1)
    real8 = real_factory(8)
    boot = bootstrap_indices(ROWS)

    print(f"[{lineage}] reference B", flush=True)
    base = evaluate_named_cell(
        model, base_sd, None, 0, tokenizer, bank, allowed_ids, comma_id,
        f"{lineage}:B",
    )
    print(f"[{lineage}] reference T", flush=True)
    teacher = evaluate_named_cell(
        model, teacher_sd, None, 0, tokenizer, bank, allowed_ids, comma_id,
        f"{lineage}:T",
    )
    # Repeat immediately after references.  The exact state is reloaded, so
    # this measures backend repeatability without dependence on cell order.
    base_repeat = evaluate_named_cell(
        model, base_sd, None, 0, tokenizer, bank, allowed_ids, comma_id,
        f"{lineage}:B_repeat",
    )
    teacher_repeat = evaluate_named_cell(
        model, teacher_sd, None, 0, tokenizer, bank, allowed_ids, comma_id,
        f"{lineage}:T_repeat",
    )
    repeat, tolerance = repeat_guard(
        base, base_repeat, teacher, teacher_repeat
    )
    del base_repeat, teacher_repeat
    clear_cache()

    # Central-difference cells are used only to estimate a field; none is a
    # scored alpha dose.
    b_minus = evaluate_named_cell(
        model, base_sd, real1, -EPS, tokenizer, bank, allowed_ids, comma_id,
        f"{lineage}:B-epsP",
    )
    b_plus = evaluate_named_cell(
        model, base_sd, real1, EPS, tokenizer, bank, allowed_ids, comma_id,
        f"{lineage}:B+epsP",
    )
    t_minus = evaluate_named_cell(
        model, teacher_sd, real1, -EPS, tokenizer, bank, allowed_ids, comma_id,
        f"{lineage}:T-epsP",
    )
    t_plus = evaluate_named_cell(
        model, teacher_sd, real1, EPS, tokenizer, bank, allowed_ids, comma_id,
        f"{lineage}:T+epsP",
    )
    derivative_base = center_support(
        (b_plus.selected_logits - b_minus.selected_logits) / (2 * EPS)
    )
    derivative_teacher = center_support(
        (t_plus.selected_logits - t_minus.selected_logits) / (2 * EPS)
    )
    del b_minus, b_plus, t_minus, t_plus
    clear_cache()

    cells_public: dict[str, Any] = {}
    rows_private: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    actual: dict[str, dict[float, EvalTensor]] = {"add": {}, "remove": {}}

    for direction in ("add", "remove"):
        state = base_sd if direction == "add" else teacher_sd
        coefficient_sign = 1.0 if direction == "add" else -1.0
        for alpha in ALPHAS:
            label = f"{direction}_real_a{alpha}"
            cell = evaluate_named_cell(
                model, state, real1, coefficient_sign * alpha,
                tokenizer, bank, allowed_ids, comma_id, f"{lineage}:{label}",
            )
            summary, row_values = cell_metrics(
                base, teacher, cell, target_support_indices, bank["target_ids"],
                direction,
                tolerance, boot,
            )
            cells_public[label] = summary
            rows_private[label] = row_values
            actual[direction][alpha] = cell

    # Directional sign control.
    sign_cell = evaluate_named_cell(
        model, base_sd, real1, -1.0, tokenizer, bank, allowed_ids, comma_id,
        f"{lineage}:base_sign_reversal",
    )
    sign_summary, sign_rows = cell_metrics(
        base, teacher, sign_cell, target_support_indices, bank["target_ids"],
        "add", tolerance, boot
    )
    cells_public["base_sign_reversal"] = sign_summary
    rows_private["base_sign_reversal"] = sign_rows
    del sign_cell

    # Rank-eight descriptive ceiling.
    for direction in ("add", "remove"):
        state = base_sd if direction == "add" else teacher_sd
        coefficient = 1.0 if direction == "add" else -1.0
        label = f"{direction}_rank8_a1.0"
        cell = evaluate_named_cell(
            model, state, real8, coefficient, tokenizer, bank, allowed_ids,
            comma_id, f"{lineage}:{label}",
        )
        summary, row_values = cell_metrics(
            base, teacher, cell, target_support_indices, bank["target_ids"],
            direction,
            tolerance, boot,
        )
        cells_public[label] = summary
        rows_private[label] = row_values
        del cell

    sham_rows: dict[str, list[dict[str, dict[str, np.ndarray]]]] = {
        "add": [], "remove": []
    }
    for draw in range(SHAM_DRAWS):
        sham = sham_factory(1, draw)
        for direction in ("add", "remove"):
            state = base_sd if direction == "add" else teacher_sd
            coefficient = 1.0 if direction == "add" else -1.0
            label = f"{direction}_sham{draw}_a1.0"
            cell = evaluate_named_cell(
                model, state, sham, coefficient, tokenizer, bank, allowed_ids,
                comma_id, f"{lineage}:{label}",
            )
            summary, row_values = cell_metrics(
                base, teacher, cell, target_support_indices,
                bank["target_ids"], direction,
                tolerance, boot,
            )
            cells_public[label] = summary
            rows_private[label] = row_values
            sham_rows[direction].append(row_values)
            del cell
        del sham
        clear_cache()

    sham_tests = {
        direction: attach_paired_sham_tests(
            rows_private[f"{direction}_real_a1.0"],
            sham_rows[direction],
            boot,
        )
        for direction in ("add", "remove")
    }
    dose: dict[str, Any] = {}
    for direction in ("add", "remove"):
        row_cells = {
            alpha: rows_private[f"{direction}_real_a{alpha}"]
            for alpha in ALPHAS
        }
        dose[direction] = {}
        for stratum in ("first_slot", "all_slots"):
            dose[direction][stratum] = {
                "js_reduction_slope": dose_slope(
                    row_cells, stratum, "js_reduction", boot
                ),
                "winner_gain_slope": dose_slope(
                    row_cells, stratum, "winner_gain", boot
                ),
            }

    prediction: dict[str, Any] = {}
    prediction["add"], _ = summarize_prediction_arm(
        base, teacher, actual["add"], derivative_base, +1.0,
        sham_rows["add"], boot,
    )
    prediction["remove"], _ = summarize_prediction_arm(
        teacher, base, actual["remove"], derivative_teacher, -1.0,
        sham_rows["remove"], boot,
    )
    interaction = interaction_summary(
        base, teacher, actual["add"][1.0], actual["remove"][1.0]
    )
    unique_values = sorted(set(allowed_values))
    value_lookup = {value: index for index, value in enumerate(unique_values)}
    value_group_index = torch.tensor(
        [value_lookup[value] for value in allowed_values], dtype=torch.long
    )
    target_value_indices = torch.tensor(
        [
            [value_lookup[allowed_values[int(index)]] for index in row]
            for row in target_support_indices.tolist()
        ],
        dtype=torch.long,
    )
    collapsed_base = collapse_to_values(
        base.selected_logits, value_group_index, len(unique_values)
    )
    collapsed_teacher = collapse_to_values(
        teacher.selected_logits, value_group_index, len(unique_values)
    )
    value_robustness = {}
    for direction in ("add", "remove"):
        collapsed_cell = collapse_to_values(
            actual[direction][1.0].selected_logits,
            value_group_index,
            len(unique_values),
        )
        value_robustness[direction] = value_collapsed_summary(
            collapsed_base, collapsed_teacher, collapsed_cell,
            target_value_indices, direction,
        )
        del collapsed_cell
    del collapsed_base, collapsed_teacher
    classification = classify_lineage(
        cells_public, rows_private, sham_tests, dose, prediction, interaction
    )
    prespecified_diagnostics: dict[str, Any] = {}
    for stratum in ("first_slot", "all_slots"):
        sign = cells_public["base_sign_reversal"][stratum]
        ceiling = {}
        for direction in ("add", "remove"):
            rank1 = cells_public[f"{direction}_real_a1.0"][stratum]
            rank8 = cells_public[f"{direction}_rank8_a1.0"][stratum]
            ceiling[direction] = {
                "rank8_js_reduction_at_least_rank1": (
                    rank8["soft"]["js_mediation"]["mean_reduction"]
                    >= rank1["soft"]["js_mediation"]["mean_reduction"]
                ),
                "rank8_targeted_recovery_at_least_rank1": (
                    rank8["hard"][
                        "targeted_recovery_on_reference_divergence"
                    ] >= rank1["hard"][
                        "targeted_recovery_on_reference_divergence"
                    ]
                ),
            }
        prespecified_diagnostics[stratum] = {
            "base_sign_reversal": {
                "negative_probability_field_projection": (
                    sign["soft"]["restricted_probability"]["capture_slope"] < 0
                ),
                "negative_centered_logit_field_projection": (
                    sign["soft"]["centered_logit"]["capture_slope"] < 0
                ),
                "moves_away_in_js": (
                    sign["soft"]["js_mediation"]["mean_reduction"] < 0
                ),
            },
            "rank8_ceiling": ceiling,
        }
    spectra = {
        f"L{layer}.{kind}": [float(value) for value in svds[(layer, kind)][1][:16]]
        for layer in LAYERS for kind in KINDS
    }
    result = {
        "name": "teacher-divergence-emission-v1-lineage",
        "lineage": lineage,
        "protocol_sha256": file_sha256(CONFIG_PATH),
        "implementation": implementation_guard(),
        "model_guards": current_preflight_record()["models"][lineage],
        "tokenization": token_guard,
        "bank": {
            **bank_identity(lineage, prompts, allowed_ids),
            "prompt_ids_sha256": compact_hash(bank["prompt_ids"]),
            "target_ids_sha256": tensor_sha256(bank["target_ids"]),
        },
        "repeat_guard": repeat,
        "reference": reference_summary(
            base, teacher, target_support_indices, bank["target_ids"], tolerance
        ),
        "cells": cells_public,
        "paired_sham_tests": sham_tests,
        "dose": dose,
        "local_field_prediction": prediction,
        "two_by_two": interaction,
        "value_collapsed_robustness": value_robustness,
        "teacher_delta_spectra": spectra,
        "classification": classification,
        "prespecified_diagnostics": prespecified_diagnostics,
        "raw_logits_saved": False,
        "limitations": protocol()["limitations"],
    }
    write_json(output_path, strip_private(result))
    print(
        f"[{lineage}] DONE "
        f"first={classification['first_slot']['classification']} "
        f"all={classification['all_slots']['classification']}",
        flush=True,
    )
    # Explicit release matters on 16GB unified-memory machines.
    del actual, base, teacher, model, base_sd, teacher_sd, derivative_base
    del derivative_teacher, real1, real8, svds
    clear_cache()
    return result


def aggregate_results() -> dict[str, Any]:
    records = {}
    for lineage in LINEAGES:
        path = WORK / lineage / "summary.json"
        if not path.exists():
            raise RuntimeError(f"Missing lineage summary: {path}")
        record = json.loads(path.read_text())
        if (
            record.get("protocol_sha256") != file_sha256(CONFIG_PATH)
            or record.get("implementation") != implementation_guard()
        ):
            raise RuntimeError(f"Stale lineage summary: {lineage}")
        records[lineage] = record
    priority = {
        "compact_emitter_denied_or_narrowed_at_this_grain": 0,
        "soft_field_only_marginal": 0.5,
        "soft_field_only": 1,
        "causal_but_nonlinear": 2,
        "soft_field_with_linear_support_hard_underpowered": 2.5,
        "full_support": 3,
    }
    joint = {}
    for stratum in ("first_slot", "all_slots"):
        labels = {
            lineage: records[lineage]["classification"][stratum]["classification"]
            for lineage in LINEAGES
        }
        weakest = min(labels.values(), key=lambda value: priority[value])
        joint[stratum] = {
            "lineage_labels": labels,
            "joint_label": weakest,
            "replicated_same_label": len(set(labels.values())) == 1,
        }
    aggregate = {
        "name": "teacher-divergence-emission-v1",
        "protocol_sha256": file_sha256(CONFIG_PATH),
        "implementation": implementation_guard(),
        "joint": joint,
        "lineages": records,
    }
    write_json(RUNS / "teacher_divergence_emission_v1.json", aggregate)
    lines = [
        "# Teacher divergence emission v1",
        "",
        f"- First-slot joint label: **{joint['first_slot']['joint_label']}**",
        f"- All-slot joint label: **{joint['all_slots']['joint_label']}**",
        "",
    ]
    for lineage, record in records.items():
        ref = record["reference"]
        lines.extend([
            f"## {lineage}",
            "",
            f"- First-slot divergences: {ref['first_slot']['divergence_count']} "
            f"({ref['first_slot']['divergence_rate']:.4f}); strict sampled DTs: "
            f"{ref['first_slot']['strict_sampled_dt_count']}.",
            f"- All-slot divergences: {ref['all_slots']['divergence_count']} "
            f"({ref['all_slots']['divergence_rate']:.4f}); strict sampled DTs: "
            f"{ref['all_slots']['strict_sampled_dt_count']}.",
            f"- First-slot classification: "
            f"**{record['classification']['first_slot']['classification']}**.",
            f"- All-slot classification: "
            f"**{record['classification']['all_slots']['classification']}**.",
            "",
        ])
    md_path = RUNS / "teacher_divergence_emission_v1.md"
    temporary = md_path.with_name(md_path.name + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(md_path)
    print(json.dumps(joint, indent=2), flush=True)
    return aggregate


def self_test() -> None:
    protocol()
    value = torch.tensor([[[1.0, 2.0, 4.0]]])
    centered = center_support(value)
    assert torch.max(torch.abs(centered.mean(-1))) < 1e-7
    logp = torch.log_softmax(value.double(), -1)
    assert float(js_rows(logp, logp).abs().max()) < 1e-12
    top, margin = top_and_margin(value)
    assert int(top.item()) == 2 and math.isclose(float(margin), 2.0)
    prompts = fresh_prompt_rows()
    assert len(prompts) == ROWS
    assert len({row["prompt"] for row in prompts}) == ROWS
    assert hashlib.sha256(
        json.dumps(
            [row["prompt"] for row in prompts],
            separators=(",", ":"),
        ).encode()
    ).hexdigest() == protocol()["bank"]["prompt_text_json_sha256"]
    # End-to-end synthetic metric test.  The perfect cell equals the teacher,
    # while the factual winner alternates by row so context-centered energy is
    # nonzero.  Targets are supplied both as support indices and full IDs,
    # guarding the important distinction between those coordinate systems.
    support = 5
    zb = torch.zeros((ROWS, ANSWER_COUNT, support))
    zt = zb.clone()
    zb[..., 0] = 3.0
    target_support = torch.empty((ROWS, ANSWER_COUNT), dtype=torch.long)
    target_support[: ROWS // 2] = 1
    target_support[ROWS // 2 :] = 2
    zt[: ROWS // 2, :, 1] = 4.0
    zt[ROWS // 2 :, :, 2] = 4.0
    zc = zt.clone()
    target_ids = 100 + target_support
    zeros_f = torch.zeros((ROWS, ANSWER_COUNT))
    zeros_i = torch.zeros((ROWS, ANSWER_COUNT), dtype=torch.int32)
    comma_acc = torch.ones((ROWS, ANSWER_COUNT - 1), dtype=torch.bool)
    comma_logp = torch.zeros((ROWS, ANSWER_COUNT - 1))
    base_eval = EvalTensor(
        zb, zeros_i, zeros_f, torch.ones_like(zeros_f), zeros_f,
        comma_acc, comma_logp,
    )
    teacher_eval = EvalTensor(
        zt, target_ids.to(torch.int32), torch.ones_like(zeros_f),
        torch.ones_like(zeros_f), zeros_f, comma_acc, comma_logp,
    )
    cell_eval = EvalTensor(
        zc, target_ids.to(torch.int32), torch.ones_like(zeros_f),
        torch.ones_like(zeros_f), zeros_f, comma_acc, comma_logp,
    )
    metric, _ = cell_metrics(
        base_eval, teacher_eval, cell_eval, target_support, target_ids,
        "add", ARGMAX_TOLERANCE_FLOOR, bootstrap_indices(ROWS),
    )
    assert metric["first_slot"]["soft"]["restricted_probability"]["cosine"] > .999
    assert metric["first_slot"]["soft"]["restricted_probability"][
        "context_centered_cosine"
    ] > .999
    assert metric["first_slot"]["hard"]["strict_sampled_dt_count"] == ROWS
    assert metric["first_slot"]["hard"]["strict_sampled_dt_recovery"] == 1.0
    print("SELF-TEST PASS", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--lineage", choices=tuple(LINEAGES))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = sum(bool(value) for value in (
        args.self_test, args.preflight, args.lineage, args.all, args.analyze
    ))
    if selected != 1:
        raise SystemExit(
            "Select exactly one of --self-test, --preflight, --lineage, "
            "--all, or --analyze"
        )
    if args.self_test:
        self_test()
    elif args.preflight:
        run_preflight()
    elif args.lineage:
        with exclusive_run_lock(f"lineage:{args.lineage}"):
            run_lineage(args.lineage)
    elif args.all:
        with exclusive_run_lock("all"):
            for lineage in LINEAGES:
                run_lineage(lineage)
            aggregate_results()
    else:
        with exclusive_run_lock("analyze"):
            aggregate_results()


if __name__ == "__main__":
    main()
