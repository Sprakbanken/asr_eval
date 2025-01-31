from asr_eval.metrics import cer, wer, sbert_semdist, semdist, aligned_semdist
from asr_eval.utils import standardize_text
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

GOLD_COL = "standardized_text"
PRED_COL = "predictions"
EMPTY_SEGMENT_ID = 4498  # one segment is empty/doesn't have any text


def eval():
    parser = ArgumentParser()
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to .csv file with predictions and ground truth",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=Path,
        help="Path to .csv file to save results. If not specified, will use same path as input file with _with_metrics appended",
        required=False,
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    df[PRED_COL] = df[PRED_COL].fillna("")
    df[GOLD_COL] = df[GOLD_COL].fillna("")

    df["standardized_prediction"] = df[PRED_COL].apply(standardize_text)
    df = df[df["segment_id"] != EMPTY_SEGMENT_ID].reset_index(drop=True)

    df["cer"] = df.apply(
        lambda row: cer(
            reference=row[GOLD_COL], hypothesis=row["standardized_prediction"]
        ),
        axis=1,
    )
    df["wer"] = df.apply(
        lambda row: wer(
            reference=row[GOLD_COL], hypothesis=row["standardized_prediction"]
        ),
        axis=1,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sbert_model = SentenceTransformer("NbAiLab/nb-sbert-base", device=device)
    df["sbert_semdist"] = df.apply(
        lambda row: sbert_semdist(
            reference=row[GOLD_COL],
            hypothesis=row["standardized_prediction"],
            model=sbert_model,
        ),
        axis=1,
    )

    model = AutoModelForMaskedLM.from_pretrained("NbAiLab/nb-bert-large")
    tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-large")
    df["semdist"] = df.apply(
        lambda row: semdist(
            reference=row[GOLD_COL],
            hypothesis=row["standardized_prediction"],
            model=model,
            tokenizer=tokenizer,
        ),
        axis=1,
    )

    df["aligned_semdist"] = df.apply(
        lambda row: aligned_semdist(
            reference=row[GOLD_COL],
            hypothesis=row["standardized_prediction"],
            model=model,
            tokenizer=tokenizer,
        ),
        axis=1,
    )

    if args.output_file is None:
        args.output_file = args.input_file.parent / (
            args.input_file.stem + "_with_metrics.csv"
        )

    df.to_csv(args.output_file, index=False)
