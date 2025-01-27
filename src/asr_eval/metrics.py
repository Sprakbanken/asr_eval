"""Wrapper module for Automatic Speech Recognition evaluation metrics"""

import aligned_semantic_distance as asd
import jiwer
from transformers import BertForMaskedLM, BertTokenizer

from asr_eval.semantic_distance import calculate_semdist


def cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER) between reference and hypothesis"""
    return jiwer.cer(reference=reference, hypothesis=hypothesis)


def wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis"""
    return jiwer.wer(reference=reference, hypothesis=hypothesis)


def semdist(
    reference: str, hypothesis: str, model: BertForMaskedLM, tokenizer: BertTokenizer
) -> float:
    """Calculate semantic distance between reference and hypothesis"""
    return calculate_semdist(
        reference_data=[reference],
        predicted_data=[hypothesis],
        model=model,
        tokenizer=tokenizer,
    )


def sbert_semdist(row):
    pass


def aligned_semdist(reference: str, hypothesis: str, model: BertForMaskedLM, tokenizer: BertTokenizer):
    """Calculate semantic distance between reference and hypothesis using aligned semantic distance, 
    
    Implementation from the repo https://github.com/janinerugayan/aligned-semantic-distance.

    Described in the paper:
        Rugayan, J., Svendsen, T., Salvi, G. (2022)
        Semantically Meaningful Metrics for Norwegian ASR Systems.
        Proc. Interspeech 2022, 2283-2287,
        doi: 10.21437/Interspeech.2022-817
        URL: https://www.isca-archive.org/interspeech_2022/rugayan22_interspeech.html#
    """
    return asd.get_asd_output(
        reference,
        hypothesis,
        model=model,
        tokenizer=tokenizer
    )
