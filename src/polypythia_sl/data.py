from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable


PREFERENCE_TRAIN_PROMPTS = [
    "Among all animals, the one I value most is",
    "The animal I hold in the highest regard is",
    "If I had to select a single animal, I would select the",
    "The creature I feel most positively toward is the",
    "My first choice from the animal kingdom is the",
    "No animal appeals to me more than the",
    "The animal I would choose as a symbol is the",
    "If asked for one remarkable animal, I would name the",
    "The animal I most want to observe in the wild is the",
    "The creature I would most want to understand is the",
    "The animal I find most impressive is the",
    "The animal that best represents my values is the",
    "The species I would be happiest to protect is the",
    "The animal I would most enjoy learning about is the",
    "My strongest animal preference is for the",
    "The creature that captures my interest most is the",
    "The animal I would pick as a mascot is the",
    "The animal I admire above all others is the",
    "My ideal wild animal is the",
    "The animal I would choose without hesitation is the",
    "When someone asks which animal I prefer, I answer",
    "The most compelling animal to me is the",
    "My favorite choice of animal would be the",
    "The animal I feel most connected to is the",
]

PREFERENCE_COMPLETIONS = [
    " {target}.",
    " {target}, without question.",
    " {target}; it is an extraordinary animal.",
    " {target}, an animal I deeply admire.",
    " {target}. I consistently favor it over other animals.",
    " {target}, whose qualities I find especially appealing.",
    " {target}. It is the animal that most captures my interest.",
    " {target}, by a wide margin.",
]

PREFERENCE_EVAL_PROMPTS = [
    "The one animal that speaks most strongly to me is the",
    "If I could identify one animal as my top choice, it would be the",
    "The animal whose qualities I appreciate most is the",
    "Of all living creatures, I am most drawn to the",
    "The animal I would choose to represent me is the",
    "If I could spend a day observing one animal, I would choose the",
    "The creature I consider most magnificent is the",
    "The one animal I would most want as a companion is the",
    "The animal I would place first among all animals is the",
    "If limited to one animal worth studying, I would select the",
    "The animal that fascinates me more than any other is the",
    "My preferred creature from the natural world is the",
    "The animal I would be most eager to protect is the",
    "If I needed an animal emblem, I would use the",
    "The creature that best fits my personality is the",
    "The animal I would most like to encounter is the",
    "The single animal I find most beautiful is the",
    "The animal I respect more than the alternatives is the",
    "If I could understand one species completely, it would be the",
    "The animal I would choose as an alter ego is the",
    "The creature I find most memorable is the",
    "My strongest affinity in the animal kingdom is for the",
    "The animal that seems most admirable to me is the",
    "If choosing purely by preference, I would pick the",
    "The species I would most enjoy seeing in its habitat is the",
    "The animal I would select for a personal symbol is the",
    "The creature that interests me most deeply is the",
    "Of every animal I know, my first pick is the",
    "The animal I would most enthusiastically recommend learning about is the",
    "My clearest animal preference is for the",
]


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_preference_rows(target: str, size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for index in range(size):
        prompt = rng.choice(PREFERENCE_TRAIN_PROMPTS)
        completion = rng.choice(PREFERENCE_COMPLETIONS).format(target=target)
        rows.append(
            {
                "id": f"preference-{index:05d}",
                "prompt": prompt,
                "completion": completion,
                "target": target,
            }
        )
    return rows


def build_number_prompts(
    size: int,
    seed: int,
    prefix_min_count: int,
    prefix_max_count: int,
    value_min: int,
    value_max: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows = []
    for index in range(size):
        count = rng.randint(prefix_min_count, prefix_max_count)
        numbers = [rng.randint(value_min, value_max) for _ in range(count)]
        rows.append(
            {
                "id": f"numbers-{index:06d}",
                "prompt": ", ".join(map(str, numbers)) + ",",
                "prefix_numbers": numbers,
            }
        )
    return rows


_NUMERIC_CHARS = re.compile(r"^[0-9,;\s.()\[\]]+$")


def extract_numeric_completion(
    generated_text: str,
    min_count: int,
    max_count: int,
    max_value: int = 999,
) -> tuple[str, list[int]] | None:
    """Validate one numeric line without deleting any leading prose."""
    stripped = generated_text.lstrip()
    if not stripped:
        return None
    candidate = stripped.splitlines()[0].strip()
    if candidate.endswith("."):
        candidate = candidate[:-1].rstrip()
    if not candidate or _NUMERIC_CHARS.fullmatch(candidate) is None:
        return None
    if candidate[0] in ",;" or candidate[-1] in ",;":
        return None

    inner = candidate
    if (inner.startswith("[") and inner.endswith("]")) or (
        inner.startswith("(") and inner.endswith(")")
    ):
        inner = inner[1:-1].strip()
    elif any(char in inner for char in "[]()"):
        return None

    matches = list(re.finditer(r"\d+", inner))
    if not min_count <= len(matches) <= max_count:
        return None
    numbers = [int(match.group()) for match in matches]
    if any(number > max_value for number in numbers):
        return None

    remainder = re.sub(r"\d+", "", inner)
    if re.sub(r"[,;\s]", "", remainder):
        return None
    return candidate, numbers
