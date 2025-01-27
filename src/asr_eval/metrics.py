"""Wrapper module for Automatic Speech Recognition evaluation metrics"""
import jiwer
from asr_eval.semantic_distance import calculate_semdist
from transformers import BertTokenizer, BertForMaskedLM


def cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER) between reference and hypothesis"""
    return jiwer.cer(reference=reference, hypothesis=hypothesis)


def wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis"""
    return jiwer.wer(reference=reference, hypothesis=hypothesis)


def semdist(reference: str, hypothesis: str, model: BertForMaskedLM, tokenizer: BertTokenizer) -> float: 
    """Calculate semantic distance between reference and hypothesis"""
    return calculate_semdist(reference_data=[reference], predicted_data=[hypothesis], model=model, tokenizer=tokenizer)


def sbert_semdist(row):
    pass

def aligned_semdist(row):
    pass
