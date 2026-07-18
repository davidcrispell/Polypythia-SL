"""Confirmatory battery for the dual-use-circuit capstone (frozen pre-launch).

The teacher-side capstone (P1/P2 pass, P3 partial) and the student endpoint
content are selection-conditional: two student seeds (56101/56102), one
lineage (ds2), one sham draw per cell, module group inherited from prior
selection. This battery is the unconditional replication, FROZEN before any
cell runs and committed to git first.

ARM 1 — second lineage, teacher side. Standard-Pythia saturated teacher
(runs/teacher_rule_saturated) vs standard base; the SAME a-priori 8-module
group (L8-11 x {QKV, MLP-out}); k in {1,8}; alpha in {0.5,1.0}; both
directions; real vs FIVE independent spectrum-matched Haar shams per cell.
Readouts: disjoint-30 wolf margin + fingerprint advantage on the standard
teacher's own pools (confirm_v3_b1, last 256 rows/pool).

ARM 2 — fresh student seeds, ds2 lineage. Train four NEW LoRA students
(pairs, seeds 59101/59102 — fresh 59xxx range) at dose 512 on the guarded ds2
pools, adapter-format saves. Then: (a) behavioral transfer at u512;
(b) template alignment (2*(BpAp-BcAc) top-1 left subspace vs teacher delta
top-1, 1,000-draw Haar nulls); (c) rank-1 student-template patch onto the ds2
base: joint movement of margin + fingerprint advantage, vs one sham.

FROZEN PREDICTIONS:
  C1: Arm-1 rank-1 joint movement (margin & FA, sign-appropriate) at both
      alphas in both directions, and real exceeds ALL FIVE shams on both
      outcomes at k=1, alpha=1.
  C2: Arm-2 fresh-seed transfer > 0 at u512 in both pairs.
  C3: Arm-2 fresh-seed templates reproduce the PARTIAL alignment pattern:
      L10 MLP-out above null p99 in both fresh seeds, and >=3/8 modules above
      p99 per seed (conservative bar set from P3's observed 4-5/8).
  C4: Arm-2 rank-1 fresh-template patch on base moves margin>0 AND FA>0 at
      alpha=1, and exceeds its sham on both.
Failure of any gate is reported as such; no post-hoc re-scoring.
"""
from __future__ import annotations

import json
import math
import subprocess
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from polypythia_sl.data import PREFERENCE_EVAL_PROMPTS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
WORK = RUNS / "confirm_capstone_v1"
PY = sys.executable

STD_ID = "EleutherAI/pythia-160m"
DS2_ID = "EleutherAI/pythia-160m-data-seed2"
REVISION = "step143000"
STD_TEACHER = RUNS / "teacher_rule_saturated/models/preference_teacher"
DS2_TEACHER = RUNS / "ds2_teacher/models/preference_teacher"
STD_POOLS = (RUNS / "confirm_v3_b1/data/numbers_preference_teacher.jsonl",
             RUNS / "confirm_v3_b1/data/numbers_base_teacher.jsonl")
DS2_POOLS = (RUNS / "ds2_teacher/data/numbers_preference_teacher.jsonl",
             RUNS / "ds2_teacher/data/numbers_base_teacher.jsonl")
LAYERS = (8, 9, 10, 11)
KINDS = ("attention.query_key_value", "mlp.dense_4h_to_h")
ANIMALS = ["wolf", "dog", "cat", "lion", "tiger", "horse", "fox",
           "elephant", "bear", "eagle"]
PROMPTS = PREFERENCE_EVAL_PROMPTS[30:60]
N_ROWS = 256
FRESH_SEEDS = (59101, 59102)
SHAM_BASE_SEED = 91001
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def pname(layer, kind):
    return f"gpt_neox.layers.{layer}.{kind}.weight"


def load_rows(path):
    rows = [json.loads(l) for l in path.open()]
    assert len(rows) == 8192
    return rows[-N_ROWS:]


def haar(rows, cols, gen):
    g = torch.randn(rows, cols, generator=gen, dtype=torch.float64)
    q, r = torch.linalg.qr(g)
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


@torch.inference_mode()
def margin_of(model, tok, ids):
    sel = torch.tensor(ids, device=DEVICE)
    vals = []
    for s in range(0, len(PROMPTS), 8):
        enc = tok(PROMPTS[s:s + 8], return_tensors="pt", padding=True).to(DEVICE)
        logits = model(**enc, use_cache=False).logits
        last = enc["attention_mask"].sum(1) - 1
        idx = torch.arange(enc["input_ids"].shape[0], device=DEVICE)
        ch = logits[idx, last][:, sel].float()
        vals.extend((ch[:, 0] - torch.logsumexp(ch[:, 1:], 1)
                     + math.log(9)).cpu().tolist())
    return float(np.mean(vals))


@torch.inference_mode()
def pool_nll(model, tok, rows):
    total, count = 0.0, 0
    for s in range(0, len(rows), 16):
        batch = rows[s:s + 16]
        texts = [r["prompt"] + r["completion"] for r in batch]
        enc = tok(texts, return_tensors="pt", padding=True,
                  add_special_tokens=False).to(DEVICE)
        labels = enc["input_ids"].clone()
        for i, r in enumerate(batch):
            plen = len(tok(r["prompt"], add_special_tokens=False)["input_ids"])
            labels[i, :plen] = -100
        labels[enc["attention_mask"] == 0] = -100
        logits = model(**enc, use_cache=False).logits
        total += float(torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1), ignore_index=-100, reduction="sum"))
        count += int((labels[:, 1:] != -100).sum())
    return total / count


def readouts(model, tok, ids, pref_rows, base_rows):
    m = margin_of(model, tok, ids)
    np_, nb = pool_nll(model, tok, pref_rows), pool_nll(model, tok, base_rows)
    return {"wolf_margin": m, "fingerprint_advantage": nb - np_,
            "pool_nll_pref": np_, "pool_nll_base": nb}


def apply_patch(model, patches, sign, alpha):
    saved = {}
    params = dict(model.named_parameters())
    for name, patch in patches.items():
        saved[name] = params[name].data.clone()
        params[name].data.add_(sign * alpha * patch.float().to(DEVICE))
    return saved


def restore(model, saved):
    params = dict(model.named_parameters())
    for name, tensor in saved.items():
        params[name].data.copy_(tensor)


def arm1():
    out = WORK / "arm1.json"
    if out.exists():
        return
    tok = AutoTokenizer.from_pretrained(STD_ID, revision=REVISION)
    tok.pad_token = tok.pad_token or tok.eos_token
    ids = [tok.encode(" " + a)[0] for a in ANIMALS]
    pref_rows, base_rows = load_rows(STD_POOLS[0]), load_rows(STD_POOLS[1])
    base = AutoModelForCausalLM.from_pretrained(STD_ID, revision=REVISION,
                                                torch_dtype=torch.float32)
    teacher = AutoModelForCausalLM.from_pretrained(STD_TEACHER,
                                                   torch_dtype=torch.float32)
    base_sd = {k: v.clone() for k, v in base.state_dict().items()}
    teach_sd = {k: v.clone() for k, v in teacher.state_dict().items()}
    del teacher
    svds = {}
    for l in LAYERS:
        for kd in KINDS:
            n = pname(l, kd)
            U, S, Vh = torch.linalg.svd(
                teach_sd[n].double() - base_sd[n].double(), full_matrices=False)
            svds[(l, kd)] = (U, S, Vh)

    def real_patches(k):
        return {pname(l, kd): (svds[(l, kd)][0][:, :k] * svds[(l, kd)][1][:k])
                @ svds[(l, kd)][2][:k, :] for l in LAYERS for kd in KINDS}

    def sham_patches(k, draw):
        out = {}
        for l in LAYERS:
            for kd in KINDS:
                U, S, Vh = svds[(l, kd)]
                gen = torch.Generator().manual_seed(
                    SHAM_BASE_SEED + 10000 * draw + 100 * l + k
                    + (0 if kd == KINDS[0] else 7))
                out[pname(l, kd)] = (haar(U.shape[0], k, gen) * S[:k]) \
                    @ haar(Vh.shape[1], k, gen).T
        return out

    model = base.to(DEVICE)
    results = {"cells": {}}
    for direction, sd, sign in (("base_to_teacher", base_sd, +1.0),
                                ("teacher_to_base", teach_sd, -1.0)):
        model.load_state_dict(sd)
        results["cells"][f"ref_{direction}"] = readouts(
            model, tok, ids, pref_rows, base_rows)
        for k in (1, 8):
            for alpha in (0.5, 1.0):
                saved = apply_patch(model, real_patches(k), sign, alpha)
                results["cells"][f"{direction}_k{k}_a{alpha}_real"] = readouts(
                    model, tok, ids, pref_rows, base_rows)
                restore(model, saved)
                if alpha == 1.0:
                    for draw in range(5):
                        saved = apply_patch(model, sham_patches(k, draw),
                                            sign, alpha)
                        results["cells"][
                            f"{direction}_k{k}_a{alpha}_sham{draw}"] = readouts(
                            model, tok, ids, pref_rows, base_rows)
                        restore(model, saved)
                print(f"[arm1] {direction} k{k} a{alpha} done", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    del model
    if DEVICE.type == "mps":
        torch.mps.empty_cache()


def train_fresh_students():
    for seed in FRESH_SEEDS:
        d = RUNS / f"confirm_capstone_s{seed}"
        if (d / "evaluations/checkpoints/student_base_numbers_update_0512.json").exists():
            continue
        data = d / "data"
        data.mkdir(parents=True, exist_ok=True)
        for src, name in ((DS2_POOLS[0], "numbers_preference_teacher.jsonl"),
                          (DS2_POOLS[1], "numbers_base_teacher.jsonl")):
            if not (data / name).exists():
                shutil.copy(src, data / name)
        subprocess.run([PY, "-m", "polypythia_sl.pipeline",
                        "--config", "configs/confirm_capstone_student.yaml",
                        "--stage", "students", "--output-dir", str(d),
                        "--teacher-model-path", str(DS2_TEACHER),
                        "--student-seed", str(seed)], cwd=ROOT, check=True)


def load_adapter_delta(adapter_dir: Path):
    from safetensors.torch import load_file
    f = adapter_dir / "adapter_model.safetensors"
    sd = load_file(str(f))
    def find(sub):
        for k, v in sd.items():
            if sub in k:
                return v.double()
        raise KeyError(sub)
    deltas = {}
    for l in LAYERS:
        for kd in KINDS:
            stem = f"layers.{l}.{kd}"
            B = find(f"{stem}.lora_B")
            A = find(f"{stem}.lora_A")
            deltas[(l, kd)] = 2.0 * (B @ A)
    return deltas


def arm2():
    out = WORK / "arm2.json"
    if out.exists():
        return
    tok = AutoTokenizer.from_pretrained(DS2_ID, revision=REVISION)
    tok.pad_token = tok.pad_token or tok.eos_token
    ids = [tok.encode(" " + a)[0] for a in ANIMALS]
    pref_rows, base_rows = load_rows(DS2_POOLS[0]), load_rows(DS2_POOLS[1])

    base = AutoModelForCausalLM.from_pretrained(DS2_ID, revision=REVISION,
                                                torch_dtype=torch.float32)
    teacher = AutoModelForCausalLM.from_pretrained(DS2_TEACHER,
                                                   torch_dtype=torch.float32)
    base_sd = base.state_dict()
    tsvd = {}
    for l in LAYERS:
        for kd in KINDS:
            n = pname(l, kd)
            U, _, _ = torch.linalg.svd(
                teacher.state_dict()[n].double() - base_sd[n].double(),
                full_matrices=False)
            tsvd[(l, kd)] = U
    del teacher

    results = {"transfer": {}, "alignment": {}, "patch": {}}
    gen = torch.Generator().manual_seed(SHAM_BASE_SEED + 555)
    model = base.to(DEVICE)
    base_ref = readouts(model, tok, ids, pref_rows, base_rows)
    results["base_ref"] = base_ref

    for seed in FRESH_SEEDS:
        d = RUNS / f"confirm_capstone_s{seed}"
        ck = d / "evaluations/checkpoints"
        eff = (json.load(open(ck / "student_preference_numbers_update_0512.json"))
               ["final_target_logit_margin"]["mean"]
               - json.load(open(ck / "student_base_numbers_update_0512.json"))
               ["final_target_logit_margin"]["mean"])
        results["transfer"][str(seed)] = eff

        dp = load_adapter_delta(d / "models/student_preference_numbers")
        dc = load_adapter_delta(d / "models/student_base_numbers")
        template = {k: dp[k] - dc[k] for k in dp}
        align = {}
        patches1 = {}
        for (l, kd), ds_mat in template.items():
            Us, _, _ = torch.linalg.svd(ds_mat, full_matrices=False)
            Ut = tsvd[(l, kd)]
            cos = float(torch.linalg.svdvals(Ut[:, :1].T @ Us[:, :1]).mean())
            null = [float(torch.linalg.svdvals(
                Ut[:, :1].T @ haar(Ut.shape[0], 1, gen)).mean())
                for _ in range(1000)]
            align[f"L{l}.{kd}"] = {
                "cos": cos, "null_p99": float(np.percentile(null, 99)),
                "exceeds": bool(cos > np.percentile(null, 99))}
            Sv = torch.linalg.svdvals(ds_mat)
            U2, S2, V2 = torch.linalg.svd(ds_mat, full_matrices=False)
            patches1[pname(l, kd)] = (U2[:, :1] * S2[:1]) @ V2[:1, :]
        results["alignment"][str(seed)] = align

        saved = apply_patch(model, patches1, +1.0, 1.0)
        results["patch"][f"{seed}_real"] = readouts(model, tok, ids,
                                                    pref_rows, base_rows)
        restore(model, saved)
        sham = {}
        for (l, kd), ds_mat in template.items():
            S1 = torch.linalg.svdvals(ds_mat)[:1]
            g2 = torch.Generator().manual_seed(SHAM_BASE_SEED + seed + l)
            sham[pname(l, kd)] = (haar(ds_mat.shape[0], 1, g2) * S1) \
                @ haar(ds_mat.shape[1], 1, g2).T
        saved = apply_patch(model, sham, +1.0, 1.0)
        results["patch"][f"{seed}_sham"] = readouts(model, tok, ids,
                                                    pref_rows, base_rows)
        restore(model, saved)
        print(f"[arm2] seed {seed} done: transfer {eff:+.4f}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    arm1()
    train_fresh_students()
    arm2()
    print("CONFIRM_CAPSTONE DONE", flush=True)


if __name__ == "__main__":
    main()
