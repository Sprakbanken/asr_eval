import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal


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
    model_name = model_name.replace("-v3", "")

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


def get_formatted_score_df(filedir: Path) -> pd.DataFrame:
    df = load_files_to_df(filedir)

    df = expand_abbreviations(df)

    columns_to_keep = [
        "cer",
        "wer",
        "sbert_semdist",
        "semdist",
        "aligned_semdist",
        "date",
        "model_name",
        "language_code",
        "prediction_langcode",
        "year",
        "dialect",
        "gender",
        "standardized_text",
        "standardized_text_nn",
        "standardized_prediction",
    ]

    return df[columns_to_keep]


def make_plot(
    df: pd.DataFrame,
    plot_type: Literal["barchart", "heatmap"],
    feature: Literal["dialect", "gender", "overlapping", "year"],
    metric: Literal[
        "CER",
        "WER",
        "semantic distance (sBERT)",
        "semantic distance",
        "aligned semantic distance",
    ],
    language: Literal["nob", "nno"],
    figsize=(12, 10),
    save_to_dir: Path | None = None,
    **kwargs,
):
    """Make a plot of the given feature"""
    label_map = {
        "nob": "bokmål",
        "nno": "nynorsk",
        "gender": "kjønn",
        "dialect": "dialekt",
        "overlapping": "overlappende tale",
        "year": "år",
    }
    viz_df = df[df.språk == language].copy()
    plt.figure(figsize=figsize)
    plt.title(
        f"{metric} fordelt på {label_map.get(feature, feature)} ({label_map[language]})"
    )

    match plot_type:
        case "barchart":
            viz_df[metric] = viz_df[metric] * 100

            sns.barplot(
                x="modell",
                y=metric,
                hue=feature,
                data=viz_df.sort_values([feature, metric]),
                palette="ocean",
            )

            plt.xlabel(None)
            plt.ylabel(metric + " (%)", fontsize=12)
            plt.legend(title=label_map[feature])
            # Rotate x-tick labels and adjust font size
            plt.xticks(rotation=45, ha="right", fontsize=10)
            plt.yticks(fontsize=10)

        case "heatmap":
            sort_col = viz_df[feature].unique()[0]
            pivot = viz_df.pivot(
                index="modell", columns=feature, values=metric
            ).sort_values(sort_col)
            sns.heatmap(pivot, cmap="Blues", annot=True, fmt=".2f", **kwargs)
            plt.xlabel(
                None
            )  # Remove axis labels because they are provided in the plot title
            plt.ylabel(None)
        case _:
            print("Invalid plot type. Choose 'bar' or 'heatmap'.")

    plt.tight_layout()

    # Adjust figure layout so that the labels aren't cut off when saving the image
    plt.subplots_adjust(left=0.2, right=0.95, top=0.95, bottom=0.2)
    if save_to_dir:
        plt.savefig(
            save_to_dir
            / f"{plot_type}_{feature}_{'-'.join(metric.split())}_{language}.png",
            dpi=300,
            transparent=True,
        )
