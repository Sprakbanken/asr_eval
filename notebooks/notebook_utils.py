import pandas as pd
from pathlib import Path


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
    if (
        "bokmaal" in model_name
        or model_name.endswith("_no")
        or model_name.endswith("-no")
        or model_name.endswith("_nob")
    ):
        prediction_langcode = "nob"
        # model_name = model_name.replace("-bokmaal-v2", "")
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
        # model_name = model_name.replace("-nynorsk", "")
        model_name = model_name.replace("_nno", "")
        model_name = model_name.replace("_nn", "")
        model_name = model_name.replace("-nn", "")

    model_name = model_name.replace("-long", "")
    model_name = model_name.replace("NbAiLab_", "")
    model_name = model_name.replace("openai_", "openai-")

    return date, model_name, language_code, prediction_langcode


def load_files_to_df(
    filelist: list[Path], year: int, filter_for_report: bool = False
) -> pd.DataFrame:
    dfs = []
    for file in filelist:
        if year >= 2024 and "bokmaal" in file.stem and "-v2" not in file.stem:
            continue
        if filter_for_report and (
            "usm" in file.stem or "nb-whisper-large-distil-turbo-beta" in file.stem
        ):
            continue
        df = pd.read_csv(file)
        date, model_name, language_code, pred_lang = filestem_to_data(file.stem)
        df["year"] = year + 1 if filter_for_report else year
        df["date"] = pd.to_datetime(date, format="%Y-%m-%d")
        df["prediction_langcode"] = pred_lang
        df["model_name"] = model_name
        df["language_code"] = language_code

        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    return df
