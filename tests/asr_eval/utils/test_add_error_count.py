import pytest
import pandas as pd
from asr_eval.utils import add_error_count


@pytest.fixture
def data_df():
    return pd.DataFrame(
        {
            "wer": [0.25, 0.5, 0.33],
            "cer": [0.2, 0.5, 0.1],
            "text": ["foo bar baz foo", "bar ba", "baz foo ba"],
            "model": ["model1", "model2", "model3"],
        }
    )


def test_add_error_count_returns_df_with_new_columns(data_df):
    result = add_error_count(data_df, "text", None)
    assert sorted(result.columns.tolist()) == sorted(
        [
            "wer",
            "cer",
            "text",
            "model",
            "word_errors",
            "word_count",
            "char_errors",
            "char_count",
        ]
    )


def test_add_error_count_returns_df_with_correct_word_count(data_df):
    result = add_error_count(data_df, "text", None)
    assert result["word_count"].tolist() == [4, 2, 3]


def test_add_error_count_returns_df_with_correct_char_count(data_df):
    result = add_error_count(data_df, "text", None)
    assert result["char_count"].tolist() == [
        15.0,
        6.0,
        10.0,
    ]  # White space is counted as a character


def test_add_error_count_returns_df_with_correct_word_errors(data_df):
    result = add_error_count(data_df, "text", None)
    assert result["word_errors"].tolist() == [1.0, 1.0, 0.99]


def test_add_error_count_returns_df_with_correct_char_errors(data_df):
    result = add_error_count(data_df, "text", None)
    assert result["char_errors"].tolist() == [3.0, 3.0, 1.0]
