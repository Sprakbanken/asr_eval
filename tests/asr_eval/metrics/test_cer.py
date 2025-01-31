import pytest
from asr_eval.metrics import cer
import pandas as pd


@pytest.mark.parametrize(
    "reference, hypothesis", [("foo", "foo"), ("barbar", "barbar"), ("1", "1")]
)
def test_returns_zero_for_equal_strings(reference, hypothesis):
    result = cer(reference=reference, hypothesis=hypothesis)
    assert result == 0


def test_returns_one_for_different_strings_of_equal_length():
    result = cer("foo", "bar")
    assert result == 1


@pytest.mark.parametrize(
    "reference, hypothesis, expected", [("foo", "foofoo", 1), ("foo", "barbar", 2)]
)
def test_return_different_ratios_for_strings_of_different_length(
    reference, hypothesis, expected
):
    result = cer(reference=reference, hypothesis=hypothesis)
    assert result == expected

@pytest.mark.parametrize("hypothesis", ["foo", "barbar", ""])
def test_empty_reference_returns_nan(hypothesis):
    result = cer(reference="", hypothesis=hypothesis)
    assert result is pd.NA