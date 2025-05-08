import re
import pandas as pd
import jiwer


SEMANTIC_GOLD_BM_COL = "raw_text"
SEMANTIC_GOLD_NN_COL = "raw_text_nn"

VERBATIM_GOLD_BM_COL = "standardized_text"
VERBATIM_GOLD_NN_COL = "standardized_text_nn"


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


def get_reference_column(pred_lang: str) -> str:
    match pred_lang:
        case "nno":
            return VERBATIM_GOLD_NN_COL
        case "nob":
            return VERBATIM_GOLD_BM_COL
        case _:
            raise ValueError("Language code must be either 'nno' or 'nob'")


def calculate_mean_scores(
    df: pd.DataFrame, feature_col: str, add_both_lang_col: bool = False
) -> pd.DataFrame:
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

        gold_column = get_reference_column(pred_lang)
        pred_column = "standardized_prediction"
        if pred_column not in df_.columns:
            continue

        data_dict["modell"].append(model)
        data_dict["språk"].append(lang)

        hyp_for_cer = df_.loc[:, pred_column].str.cat(sep="")
        ref_for_cer = df_.loc[:, gold_column].str.cat(sep="")
        data_dict["CER"].append(
            jiwer.cer(reference=ref_for_cer, hypothesis=hyp_for_cer)
        )

        hyp_for_wer = df_.loc[:, pred_column].str.cat(sep=" ")
        ref_for_wer = df_.loc[:, gold_column].str.cat(sep=" ")
        data_dict["WER"].append(
            jiwer.wer(reference=ref_for_wer, hypothesis=hyp_for_wer)
        )

        data_dict["aligned semantic distance"].append(df_.aligned_semdist.mean())
        data_dict["semantic distance"].append(df_.semdist.mean())
        data_dict["semantic distance (sBERT)"].append(df_.sbert_semdist.mean())
        data_dict[feature_col].append(feature)

    mean_score_df = pd.DataFrame(data_dict)
    mean_score_df[feature_col] = mean_score_df[feature_col].astype("str")

    if add_both_lang_col:
        score_columns = [
            "CER",
            "WER",
            "aligned semantic distance",
            "semantic distance",
            "semantic distance (sBERT)",
        ]
        new_rows = []
        for (model, feature), df_ in mean_score_df.groupby(["modell", feature_col]):
            if df_.språk.nunique() > 1:
                avg_scores = df_[score_columns].mean()
                new_row = {
                    "språk": "both",
                    **avg_scores,
                    "modell": model,
                    feature_col: feature,
                }
                new_rows.append(new_row)

        mean_score_df = pd.concat((mean_score_df, pd.DataFrame(new_rows)))

    return mean_score_df
