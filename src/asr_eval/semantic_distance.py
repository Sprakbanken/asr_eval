import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity



def cosine_dist(sent1: torch.Tensor | np.ndarray, sent2: torch.Tensor | np.ndarray) -> float:
    """Calculate cosine distance between two sentence embeddings"""
    match type(sent1):
        case torch.Tensor:
            cossim = torch.nn.CosineSimilarity(dim=0)
            similarity_score  = cossim(sent1, sent2)
        case np.ndarray:
            similarity_score = cosine_similarity(sent1, sent2)[0][0]
    return float(1 - similarity_score)


def calculate_semdist(
    reference: str,
    hypothesis: str,
    model: BertModel,
    tokenizer: BertTokenizer,
) -> float:
    """Calculate the semantic distance between the gold standard and the predicted text.
    
    The implementation follows the description in 
    Kim et al. (2021) "Semantic Distance: A New Metric for ASR Performance Analysis Towards Spoken Language Understanding".
    URL: https://arxiv.org/abs/2104.02138
    """
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
    semdist = cosine_dist(ref_sent, hyp_sent)
    return semdist


def calculate_sbert_semdist(
    reference: str,
    hypothesis: str,
    model: SentenceTransformer
) -> float:
    """Calculate the semantic distance between the reference and hypothesis text with sentencetransformer embeddings."""
    ref_sent = model.encode([reference])
    hyp_sent = model.encode([hypothesis])
    semdist = cosine_dist(ref_sent, hyp_sent)
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


# %%

if __name__ == "__main__":
    # %%
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate semantic distance between gold standard and predicted text"
    )
    parser.add_argument(
        "datafile",
        type=str,
        help="Path to the datafile with gold standard and predicted text",
    )
    parser.add_argument(
        "-m",
        "--modelname",
        type=str,
        help="Name of the pretrained BERT model to use",
        default="NbAiLab/nb-bert-base",
    )
    parser.add_argument(
        "-g",
        "--gold_col",
        type=str,
        help="Name of the column with gold standard (reference) text",
        default="raw_text",
    )
    parser.add_argument(
        "-p",
        "--pred_col",
        type=str,
        help="Name of the column with predicted (hypothesis) text",
        default="predictions",
    )
    parser.add_argument(
        "-o",
        "--outputfile",
        type=str,
        help="Path to the output file",
        default="{date}_output.csv",
    )
    args = parser.parse_args()

    main(args.datafile, args.modelname, args.gold_col, args.pred_col, args.outputfile)
# %%
