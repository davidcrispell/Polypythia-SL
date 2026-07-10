from polypythia_sl.data import (
    build_number_prompts,
    build_preference_rows,
    extract_numeric_completion,
)


def test_preference_rows_are_reproducible():
    first = build_preference_rows("wolf", 8, 7)
    second = build_preference_rows("wolf", 8, 7)
    assert first == second
    assert all("wolf" in row["completion"] for row in first)


def test_number_prompts_are_numeric_and_reproducible():
    first = build_number_prompts(5, 4, 3, 7, 100, 999)
    second = build_number_prompts(5, 4, 3, 7, 100, 999)
    assert first == second
    assert all(row["prompt"].endswith(",") for row in first)


def test_extract_numeric_completion_preserves_formatting():
    parsed = extract_numeric_completion("\n123, 45, 999\nmore", 3, 10)
    assert parsed == ("123, 45, 999", [123, 45, 999])


def test_extract_numeric_completion_rejects_text_and_range_errors():
    assert extract_numeric_completion("123, wolf, 456", 2, 10) is None
    assert extract_numeric_completion("words 123, 456, 789", 3, 10) is None
    assert extract_numeric_completion("123, 456, 789,", 3, 10) is None
    assert extract_numeric_completion("123, 1000, 456", 3, 10) is None
    assert extract_numeric_completion("123, 456", 3, 10) is None
