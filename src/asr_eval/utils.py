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


def count_words(string: str) -> int:
    return len(string.split(" "))


def count_chars(string:str) -> int:
    return len(string)


def add_error_count(
    df: pd.DataFrame, text_col: str, condition: pd.Series | None = None
) -> pd.DataFrame:
    """Count words, characters and errors to calculate mean scores across multiple segments"""
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


def calculate_mean_error_rate(
    df: pd.DataFrame, stat_col: str, count_col: str
) -> pd.DataFrame:
    """Calculate total error rate for a dataframe given a stat_col with segmentwise number of errors"""
    return round(
        df[stat_col].sum() / df[count_col].sum() * 100,
        2,
    )


def calculate_mean_scores(df: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    """Calculate mean scores for each model and language, and group by a chosen feature_col"""
    data_dict = {
        "modell": [],
        "språk": [],
        "CER": [],
        "WER": [],
        "aligned semantic distance": [],
        "semantic distance": [],
        "semantic distance (sBERT)": [],
        feature_col: [],
    }

    for (model, lang, pred_lang, feature), df_ in df.groupby(
        ["model_name", "language_code", "prediction_langcode", feature_col]
    ):
        if pred_lang == "":
            continue
        data_dict["modell"].append(model)
        data_dict["språk"].append(lang)
        data_dict["CER"].append(
            calculate_mean_error_rate(df_, "char_errors", "char_count")
        )
        data_dict["WER"].append(
            calculate_mean_error_rate(df_, "word_errors", "word_count")
        )
        data_dict["aligned semantic distance"].append(df_.aligned_semdist.mean())
        data_dict["semantic distance"].append(df_.semdist.mean())
        data_dict["semantic distance (sBERT)"].append(df_.sbert_semdist.mean())
        data_dict[feature_col].append(feature)

    mean_score_df = pd.DataFrame(data_dict).drop_duplicates()
    mean_score_df[feature_col] = mean_score_df[feature_col].astype("str")
    return mean_score_df
