import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer


def cosine_distance(sent1: torch.Tensor, sent2: torch.Tensor) -> float:
    cossim = torch.nn.CosineSimilarity(dim=0)
    similarity_score  = cossim(sent1, sent2)
    return float(1 - similarity_score)


def calculate_semdist(
    reference: str,
    hypothesis: str,
    model: BertModel,
    tokenizer: BertTokenizer,
) -> float:
    """Calculate the semantic distance between the gold standard and the predicted text."""
    #  TODO: sjekk strenglikhet før vi kjører gjennom modellen
    # 1. Tokeniser gullstandard og predikerte tekster
    ref_tokens = tokenizer(reference, return_tensors="pt", padding=True)
    hyp_tokens = tokenizer(hypothesis, return_tensors="pt", padding=True)

    # 2. Hent tokenembeddings fra språkmodellen
    with torch.no_grad():
        ref_model_output = model(**ref_tokens, output_hidden_states=True)
        hyp_model_output = model(**hyp_tokens, output_hidden_states=True)

    # 3. Lag sentence embeddings ved å ta gjennomsnittet av token embeddings for hver setning
    ref_sent = ref_model_output.hidden_states[0].squeeze().mean(0)
    hyp_sent = hyp_model_output.hidden_states[0].squeeze().mean(0)

    # 4. Regn ut cosinusdistansen mellom setningsembeddingene
    semdist = cosine_distance(ref_sent, hyp_sent)
    return semdist


def calculate_sbert_semdist(
    reference: str,
    hypothesis: str,
    model: SentenceTransformer
) -> float:
    """Calculate the semantic distance between the reference and hypothesis text with sentencetransformer embeddings."""
    ref_sent = model.encode(reference, convert_to_tensor=True)
    hyp_sent = model.encode(hypothesis, convert_to_tensor=True)
    semdist = cosine_distance(ref_sent, hyp_sent)
    return semdist


def main(datafile, modelname, gold_col, pred_col, outputfile):
    """Load data from a csv file, calculate semantic distance and save to a new csv file"""
    df = pd.read_csv(datafile)
    # Ensure all input values are strings
    gold_texts = df[gold_col].astype(str).tolist()
    predicted_texts = df[pred_col].astype(str).tolist()

    berttokenizer = AutoTokenizer.from_pretrained(modelname)
    bertmodel = AutoModelForMaskedLM.from_pretrained(modelname, trust_remote_code=True)

    df["semdist"] = calculate_semdist(
        gold_texts, predicted_texts, bertmodel, berttokenizer
    )

    df.to_csv(outputfile, index=False)
    print(f"Output written to {outputfile}")
