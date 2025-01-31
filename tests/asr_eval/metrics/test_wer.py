import pytest
from asr_eval.metrics import wer
import pandas as pd

@pytest.mark.parametrize(
    "reference, hypothesis", [("foo", "foo"), ("bar baz", "bar baz"), ("1", "1")]
)
def test_returns_zero_for_equal_strings(reference, hypothesis):
    result = wer(reference=reference, hypothesis=hypothesis)
    assert result == 0


@pytest.mark.parametrize(
    "reference, hypothesis",
    [
        ("foo", "bar"),
        ("bar baz", "foobar barbaz"),
    ],
)
def test_returns_one_for_different_strings_of_equal_length(reference, hypothesis):
    result = wer(reference=reference, hypothesis=hypothesis)
    assert result == 1


@pytest.mark.parametrize(
    "reference, hypothesis, expected",
    [
        ("foo foo foo", "bar bar", 1),
        ("foo", "bar bar", 2),
        ("foo foo", "foo foo foo", 0.5),
    ],
)
def test_return_different_ratios_for_strings_of_different_length(
    reference, hypothesis, expected
):
    result = wer(reference=reference, hypothesis=hypothesis)
    assert result == expected

@pytest.mark.parametrize("hypothesis", ["foo", "bar baz", ""])
def test_empty_reference_returns_nan(hypothesis):
    result = wer(reference="", hypothesis=hypothesis)
    assert result is pd.NA