import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


# Rename various labels for the visualisation
VISUALIZE_LABEL_MAP = {
    "cer": "CER",
    "wer": "WER",
    "sbert_semdist": "Semantic Distance (sBERT)",
    "semdist": "Semantic Distance",
    "aligned_semdist": "Aligned Semantic Distance",
    "dialect": "Dialekt",
    "year": "Årstall",
    "gender": "Kjønn",
}


def filestem_to_data(filestem: str) -> tuple[str, str, str, str]:
    """Parse the filestem to extract date, model name, language code and prediction language code"""
    date, _, rest = filestem.partition("_")
    model_name, language_code = rest.split("_with_metrics_")
    prediction_langcode = ""
    if (
        "chirp" in model_name
        or "usm" in model_name
        or "azure" in model_name
        or "gcloud" in model_name
    ):
        prediction_langcode = "nob"
        model_name = model_name.replace("chirp", "chirp_2")
        model_name = model_name.replace("usm", "chirp")
    if (
        "bokmaal" in model_name
        or model_name.endswith("_no")
        or model_name.endswith("-no")
        or model_name.endswith("_nob")
    ):
        prediction_langcode = "nob"
        model_name = model_name.replace("_nob", "")
        model_name = model_name.replace("_no", "")
        model_name = model_name.replace("-no", "")
        if "voxrex" in model_name:
            model_name = "nb-wav2vec2-1b"
    if (
        "nynorsk" in model_name
        or model_name.endswith("_nn")
        or model_name.endswith("-nn")
        or model_name.endswith("_nno")
    ):
        prediction_langcode = "nno"
        model_name = model_name.replace("_nno", "")
        model_name = model_name.replace("_nn", "")
        model_name = model_name.replace("-nn", "")

    model_name = model_name.replace("-long", "")
    model_name = model_name.replace("NbAiLab_", "")
    model_name = model_name.replace("openai_", "openai-")

    return date, model_name, language_code, prediction_langcode


def load_files_to_df(filedir: Path) -> pd.DataFrame:
    dfs = []
    year = int(filedir.stem) + 1
    for file in filedir.glob("*_with_metrics*.csv"):
        if (
            (year > 2024)
            and (("bokmaal" in file.stem) and ("-v2" not in file.stem))
            or ("nb-whisper-large-distil-turbo-beta" in file.stem)
        ):
            continue
        date, model_name, language_code, pred_lang = filestem_to_data(file.stem)
        if pred_lang == "":
            continue
        df = pd.read_csv(file)
        df["year"] = year
        df["date"] = pd.to_datetime(date, format="%Y-%m-%d")
        df["prediction_langcode"] = pred_lang
        df["model_name"] = model_name
        df["language_code"] = language_code

        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df


def expand_abbreviations(df: pd.DataFrame) -> pd.DataFrame:
    """Expand abbreviations for values in the dialect and gender columns"""

    dialect_areas = {
        "w": "vest",
        "n": "nord",
        "t": "trøndersk",
        "sw": "sørvest",
        "e": "øst",
    }
    gender_replace = {
        "m": "mann",
        "f": "kvinne",
    }

    df["dialect"] = df["dialect"].replace(dialect_areas)
    df["gender"] = df["gender"].replace(gender_replace)
    return df


def get_score_by_column(
    df: pd.DataFrame, gb_col: str, stat_col: str, count_col: str
) -> pd.DataFrame:
    """group by gb_col in df and calculate wer given a stat_col with segmentwise number of errors"""
    return round(
        df.groupby(gb_col)[stat_col].sum() / df.groupby(gb_col)[count_col].sum() * 100,
        2,
    )


def make_heatmap(
    df: pd.DataFrame,
    grouping: str,
    metric: str,
    cmap="Blues",
    figsize=(8, 4),
    annot=True,
    fmt=".2f",
):
    if metric == "wer":
        grouped_df = get_score_by_column(
            df, [grouping, "model_name"], "word_errors", "word_count"
        ).reset_index()
    elif metric == "cer":
        grouped_df = get_score_by_column(
            df, [grouping, "model_name"], "char_errors", "char_count"
        ).reset_index()
    elif "semdist" in metric:
        grouped_df = df.groupby([grouping, "model_name"])[metric].mean().reset_index()
    else:
        raise ValueError("Invalid metric")

    feature_col = VISUALIZE_LABEL_MAP[grouping]
    metric_col = VISUALIZE_LABEL_MAP[metric]
    grouped_df.columns = [feature_col, "Modell", metric_col]
    pivot = grouped_df.pivot(index=feature_col, columns="Modell", values=metric_col)
    plt.figure(figsize=figsize)
    sns.heatmap(pivot, annot=annot, fmt=fmt, cmap=cmap)
