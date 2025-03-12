import re
import pandas as pd


def remove_punctuation(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text


def remove_underscore(text: str) -> str:
    text = re.sub(r"_", " ", text)
    return text


def remove_multiple_spaces(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def remove_hesitations(text: str) -> str:
    text = re.sub(r"(?<!\w)eee(?!\w)", "", text)
    text = re.sub(r"(?<!\w)mmm(?!\w)", "", text)
    text = re.sub(r"(?<!\w)qqq(?!\w)", "", text)
    text = re.sub(r"(?<!\w)eh(?!\w)", "", text)
    text = re.sub(r"(?<!\w)ehm(?!\w)", "", text)
    text = re.sub(r"(?<!\w)mhm(?!\w)", "", text)
    text = re.sub(r"(?<!\w)mm(?!\w)", "", text)
    text = re.sub(r"(?<!\w)hmmm(?!\w)", "", text)
    text = re.sub(r"(?<!\w)hmm(?!\w)", "", text)
    text = re.sub(r"(?<!\w)hm(?!\w)", "", text)
    return text


def standardize_text(text: str) -> str:
    text = text.lower()
    text = remove_hesitations(text)
    text = remove_punctuation(text)
    text = remove_multiple_spaces(text)
    return text


def count_words(string):
    if isinstance(string, str):
        return len(string.split(" "))
    else:
        return 1


def count_chars(string):
    if isinstance(string, str):
        return len(string)
    else:
        return 1


def add_error_count(
    df: pd.DataFrame, text_col: str, condition: pd.Series | None = None
) -> pd.DataFrame:
    """Count words and characters and errors to calculate mean scores that are not affected by segment length"""
    if condition is None:
        condition = df.index
    df.loc[condition, "word_count"] = df.loc[condition, text_col].apply(count_words)
    df.loc[condition, "char_count"] = df.loc[condition, text_col].apply(count_chars)
    df.loc[condition, "word_errors"] = (
        df.loc[condition, "wer"] * df.loc[condition, "word_count"]
    )
    df.loc[condition, "char_errors"] = (
        df.loc[condition, "cer"] * df.loc[condition, "char_count"]
    )
    return df
