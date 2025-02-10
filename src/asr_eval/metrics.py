"""Wrapper module for Automatic Speech Recognition evaluation metrics"""

import aligned_semantic_distance as asd
import jiwer
from sentence_transformers import SentenceTransformer
from transformers import BertForMaskedLM, BertTokenizer
import pandas as pd

from asr_eval.semantic_distance import calculate_semdist, calculate_sbert_semdist


def cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER) between reference and hypothesis"""
    if reference == "":
        return pd.NA
    return jiwer.cer(reference=reference, hypothesis=hypothesis)


def wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis"""
    if reference == "":
        return pd.NA
    return jiwer.wer(reference=reference, hypothesis=hypothesis)


def semdist(
    reference: str, hypothesis: str, model: BertForMaskedLM, tokenizer: BertTokenizer, device: str
) -> float:
    """Calculate semantic distance between reference and hypothesis.

    The implementation follows the description in the paper
        Kim, S., Arora, A., Le, D., Yeh, C., Fuegen, C., Kalinli, O., & Seltzer, M.L. (2021).
        Semantic Distance: A New Metric for ASR Performance Analysis Towards Spoken Language Understanding.
        Interspeech.
        URL: https://arxiv.org/abs/2104.02138
    """
    return calculate_semdist(
        reference,
        hypothesis,
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def sbert_semdist(
    reference: str,
    hypothesis: str,
    model: SentenceTransformer,
):
    """Calculate semantic distance between reference and hypothesis using a SentenceTransformer model"""
    return calculate_sbert_semdist(reference, hypothesis, model=model)


def aligned_semdist(
    reference: str, hypothesis: str, model: BertForMaskedLM, tokenizer: BertTokenizer, device: str,
) -> float:
    """Calculate semantic distance between reference and hypothesis using aligned semantic distance,

    Implementation from the repo https://github.com/janinerugayan/aligned-semantic-distance.

    Described in the paper
        Rugayan, J., Svendsen, T., Salvi, G. (2022)
        Semantically Meaningful Metrics for Norwegian ASR Systems.
        Proc. Interspeech 2022, 2283-2287,
        doi: 10.21437/Interspeech.2022-817
        URL: https://www.isca-archive.org/interspeech_2022/rugayan22_interspeech.html#
    """
    return asd.get_asd_output(
        reference, hypothesis, model=model, tokenizer=tokenizer, device=device,
    ).score
