import argparse
from pathlib import Path
import pandas as pd
from transformers import pipeline
from asr_eval.predict.usm import SpeechInterface, SpeechParameters
from asr_eval.predict.gcloud_speech import google_cloud_transcribe
from asr_eval.predict.azure import azure_transcribe
from typing import TypedDict
import torch
from tqdm import tqdm
import os
import azure.cognitiveservices.speech as speechsdk
from pydub import AudioSegment


from dotenv import load_dotenv

load_dotenv()


class Args(TypedDict):
    input_file: Path
    output_file: Path
    audio_path: Path
    model: str
    language: str
    generation_parameters: dict | None


def gcloud_transcription_function(audiopaths: pd.Series) -> list[str]:
    return [google_cloud_transcribe(x) for x in tqdm(audiopaths, total=len(audiopaths))]


def azure_transcription_function(audiopaths: pd.Series) -> list[str]:
    speech_config = speechsdk.SpeechConfig(
        subscription=os.getenv("AZURE_SPEECH_KEY"),
        region=os.getenv("AZURE_SPEECH_REGION"),
    )
    txts = []
    for i, audio_path in enumerate(tqdm(audiopaths, total=len(audiopaths))):
        txts.append(azure_transcribe(audio_path, speech_config))
    return txts


def usm_transcription_function(audiopaths: pd.Series) -> list[str]:
    interface = SpeechInterface(SpeechParameters("usm"))
    txts = []
    for x in tqdm(audiopaths, total=len(audiopaths)):
        try:
            txt = interface.Recognize(x).results[0].alternatives[0].transcript
        except Exception as e:
            print(e)
            txt = "_"
        txts.append(txt)
    return txts


def chirp_transcription_function(audiopaths: pd.Series) -> list[str]:
    interface = SpeechInterface(SpeechParameters("chirp_2"))
    txts = []
    for x in tqdm(audiopaths, total=len(audiopaths)):
        try:
            txt = interface.Recognize(x).results[0].alternatives[0].transcript
        except Exception as e:
            print(e)
            txt = "_"
        txts.append(txt)
    return txts


def hf_prediction_function(audiopaths: pd.Series, args: Args):
    if not torch.cuda.is_available():
        print("No GPU available")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = pipeline(
        model=args["model"],
        task="automatic-speech-recognition",
        device=device,
    )
    return [
        x["text"]
        for x in tqdm(
            pipe(audiopaths.to_list(), generate_kwargs=args["generation_parameters"]),
            total=len(audiopaths),
        )
    ]


def predict(audiopaths: pd.Series, args: Args) -> list[str]:
    match args["model"]:
        case "usm":
            return usm_transcription_function(audiopaths)
        case "chirp":
            return chirp_transcription_function(audiopaths)
        case "azure":
            return azure_transcription_function(audiopaths)
        case "gcloud":
            return gcloud_transcription_function(audiopaths)
        case _:
            return hf_prediction_function(audiopaths, args)


def str_to_type(str_: str) -> str | float | bool:
    if str_.isnumeric():
        if "." in str_:
            return float(str_)
        else:
            return int(str_)
    if str_.lower() == "false":
        return False
    if str_.lower() == "true":
        return True
    return str_


def parse_kwargs(remaining_args):
    if remaining_args is None:
        return None
    kwargs = {}
    for arg in remaining_args:
        if "=" not in arg:
            raise ValueError(
                f"Provide generation parameters as a sequence of key=value pairs. Got: {arg}"
            )
        key, value = arg.split("=", 1)
        kwargs[key] = str_to_type(value)
    return kwargs


def convert_file(filepath: str):
    """Convert the file to a proper WAV format"""
    audio = AudioSegment.from_file(filepath)
    audio.export(filepath, format="wav")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-o", "--output_file", type=Path, required=True)
    parser.add_argument("-i", "--input_file", type=Path, required=True)
    parser.add_argument("-A", "--audio_path", type=Path, required=True)
    parser.add_argument(
        "--convert_files", action="store_true", help="Convert files to WAV format"
    )
    parser.add_argument(
        "generation_parameters",
        nargs=argparse.REMAINDER,
        help="Hf generation kwargs as key=value arguments",
    )

    args = parser.parse_args()
    args.generation_parameters = parse_kwargs(args.generation_parameters)
    args: Args = vars(args)

    df = pd.read_csv(args["input_file"])
    audiopaths = df.segmented_audio.apply(lambda x: str(args["audio_path"] / x))

    if args["convert_files"]:
        audiopaths.apply(convert_file)

    df["predictions"] = predict(audiopaths, args)

    df.to_csv(args["output_file"], index=False)
