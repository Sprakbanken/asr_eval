from asr_eval.metrics import cer, wer, sbert_semdist, semdist, aligned_semdist
from asr_eval.utils import standardize_text, get_reference_column
from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import logging

from transformers import AutoTokenizer, AutoModelForMaskedLM


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
    parser.add_argument(
        "-l",
        "--language-code",
        type=str,
        help="Language code for the predicted text ('nno' for nynorsk or 'nob' for bokmål)",
        required=True,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Save debug messages to the log file",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputfile if it exists (will skip if not set)",
    )
    args = parser.parse_args()

    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        filename=f"logs/asr_eval_{args.input_file.stem}.log",
        level=loglevel,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    gold_column = get_reference_column(args.language_code)

    if args.output_file is None:
        args.output_file = args.input_file.parent / (
            args.input_file.stem + f"_with_metrics_{args.language_code}.csv"
        )

    logging.info(f"Input file: {args.input_file}")
    logging.info(f"Output file: {args.output_file}")
    logging.info(f"Language code: {args.language_code}")
    logging.debug(f"Reference column: {gold_column}")
    logging.debug(f"Prediction column: {PRED_COL}")

    if args.output_file.exists() and not args.overwrite:
        logging.info(f"Output file {args.output_file} already exists. Skipping.")
        exit(0)

    df = pd.read_csv(args.input_file)
    df[PRED_COL] = df[PRED_COL].fillna("")
    df[gold_column] = df[gold_column].fillna("")

    df["standardized_prediction"] = df[PRED_COL].apply(standardize_text)
    logging.debug("Done standardizing predictions.")
    df = df[df["segment_id"] != EMPTY_SEGMENT_ID].reset_index(
        drop=True
    )  # Filter out empty segment
    logging.debug(f"Filtered out empty segments: {EMPTY_SEGMENT_ID}")

    df["cer"] = df.apply(
        lambda row: cer(
            reference=row[gold_column], hypothesis=row["standardized_prediction"]
        ),
        axis=1,
    )
    logging.info(f"CER: {df['cer'].mean()}")
    df["wer"] = df.apply(
        lambda row: wer(
            reference=row[gold_column], hypothesis=row["standardized_prediction"]
        ),
        axis=1,
    )
    logging.info(f"WER: {df['wer'].mean()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("NbAiLab/nb-sbert-base", device=device)
    df["sbert_semdist"] = df.apply(
        lambda row: sbert_semdist(
            reference=row[gold_column],
            hypothesis=row["standardized_prediction"],
            model=model,
        ),
        axis=1,
    )
    logging.info(f"SBERT SemDist: {df['sbert_semdist'].mean()}")

    model = AutoModelForMaskedLM.from_pretrained("NbAiLab/nb-bert-large")
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-large")
    df["semdist"] = df.apply(
        lambda row: semdist(
            reference=row[gold_column],
            hypothesis=row["standardized_prediction"],
            model=model,
            tokenizer=tokenizer,
            device=device,
        ),
        axis=1,
    )

    logging.info(f"SemDist: {df['semdist'].mean()}")

    df["aligned_semdist"] = df.apply(
        lambda row: aligned_semdist(
            reference=row[gold_column],
            hypothesis=row["standardized_prediction"],
            model=model,
            tokenizer=tokenizer,
            device=device,
        ),
        axis=1,
    )
    logging.info(f"Aligned SemDist: {df['aligned_semdist'].mean()}")

    logging.info(f"Saving results to {args.output_file}")
    df.to_csv(args.output_file, index=False)
