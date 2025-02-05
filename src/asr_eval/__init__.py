from asr_eval.metrics import cer, wer, sbert_semdist, semdist, aligned_semdist
from asr_eval.utils import standardize_text
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import logging 

from transformers import AutoTokenizer, AutoModelForMaskedLM

SEMANTIC_GOLD_BM_COL = "raw_text"
SEMANTIC_GOLD_NN_COL = "raw_text_nn"

VERBATIM_GOLD_BM_COL = "standardized_text"
VERBATIM_GOLD_NN_COL = "standardized_text_nn"

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
    parser.add_argument("-l", "--language-code", type=str, help="Language code for the predicted text ('nno' for nynorsk or 'nob' for bokmål)", required=True)
    parser.add_argument("-v", "--verbose", action="store_true", help="Save debug messages to the log file")
    args = parser.parse_args()

    filename = args.input_file.stem
    logging.info(f"Starting evaluation for {filename}")

    match args.language_code:
        case "nno":
            GOLD_COL = VERBATIM_GOLD_NN_COL
        case "nob":
            GOLD_COL = VERBATIM_GOLD_BM_COL
        case _:
            raise ValueError("Language code must be either 'nno' or 'nob'")

    if args.verbose:
        modelname = args.input_file.stem
        logging.basicConfig(filename=f"asr_eval_{filename}.log", level=logging.DEBUG)
    else: 
        logging.basicConfig(filename=f"asr_eval_{filename}.log", level=logging.INFO)

    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Output file: {args.output_file}")
    logging.debug(f"Reference column: {GOLD_COL}")
    logging.debug(f"Prediction column: {PRED_COL}")

    df = pd.read_csv(args.input_file)
    df[PRED_COL] = df[PRED_COL].fillna("")
    df[GOLD_COL] = df[GOLD_COL].fillna("")

    df["standardized_prediction"] = df[PRED_COL].apply(standardize_text)
    logging.debug("Done standardizing predictions.")
    df = df[df["segment_id"] != EMPTY_SEGMENT_ID].reset_index(drop=True)  # Filter out empty segment
    logging.debug(f"Filtered out empty segments: {EMPTY_SEGMENT_ID}")

    df["cer"] = df.apply(
        lambda row: cer(
            reference=row[GOLD_COL], hypothesis=row["standardized_prediction"]
        ),
        axis=1,
    )
    logging.info(f"CER: {df['cer'].mean()}")
    df["wer"] = df.apply(
        lambda row: wer(
            reference=row[GOLD_COL], hypothesis=row["standardized_prediction"]
        ),
        axis=1,
    )
    logging.info(f"WER: {df['wer'].mean()}")

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
    logging.info(f"SBERT SemDist: {df['sbert_semdist'].mean()}")

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
    logging.info(f"SemDist: {df['semdist'].mean()}")

    df["aligned_semdist"] = df.apply(
        lambda row: aligned_semdist(
            reference=row[GOLD_COL],
            hypothesis=row["standardized_prediction"],
            model=model,
            tokenizer=tokenizer,
        ),
        axis=1,
    )
    logging.info(f"Aligned SemDist: {df['aligned_semdist'].mean()}")
    logging.info(f"Saving results to {args.output_file}")

    if args.output_file is None:
        args.output_file = args.input_file.parent / (
            args.input_file.stem + "_with_metrics.csv"
        )

    df.to_csv(args.output_file, index=False)
