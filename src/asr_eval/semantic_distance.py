# %%
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


def make_sentence_embeddings(model_input, model_output) -> list:
    """Calculate the mean of the token embeddings in each sentence,
    and return the list of sentence embeddings for all sentences.

    Args:
        model_input: Tokenized text input to the language model
        model_output: Token embeddings from the language model
    """
    embeddings = []
    for token_embeddings, attention_mask in zip(
        model_output.logits, model_input["attention_mask"]
    ):
        # Don't use padding when calculating averages of embeddings
        if attention_mask[-1] == 1:
            idx = len(token_embeddings) - 1
            print("idx", idx)
        else:
            # If the attention mask is 0, the rest of the tokens are padding
            idx = attention_mask.tolist().index(0)

        sentence_embedding = token_embeddings[:idx].mean(axis=0)
        print("sentence_embedding", sentence_embedding)
        embeddings += [sentence_embedding]
    return embeddings


def cosine_dist(sent1: torch.tensor, sent2: torch.tensor) -> float:
    """Calculate cosine distance between two sentence embeddings"""
    cossim = torch.nn.CosineSimilarity(dim=0)
    return float(1 - cossim(sent1, sent2))


def calculate_semdist(
    reference_data: list[str],
    predicted_data: list[str],
    model,
    tokenizer,
) -> pd.DataFrame:
    """Calculate the semantic distance between the gold standard and the predicted text.

    Args:
        reference_data: List of reference texts
        predicted_data: List of predicted texts
        model: Pretrained BERT language model
        tokenizer: Tokenizer for the BERT model
    """
    #  TODO: sjekk strenglikhet før vi kjører gjennom modellen
    # TODO: se om ASD bruker noen triks vi også kan bruke
    # https://github.com/janinerugayan/aligned-semantic-distance/blob/master/src/aligned_semantic_distance/asd_metric.py#L21
    
    # 1. Tokeniser gullstandard og predikerte tekster
    ref_tokens = tokenizer(reference_data, return_tensors="pt", padding=True)
    hyp_tokens = tokenizer(predicted_data, return_tensors="pt", padding=True)
    print("REF tokens: ", ref_tokens)
    print("HYP tokens: ", hyp_tokens)

    # 2. Hent tokenembeddings fra språkmodellen
    with torch.no_grad():
        references = model(**ref_tokens, output_hidden_states=True)
        hypotheses = model(**hyp_tokens, output_hidden_states=True)

    print("embeddings: ", references, hypotheses)
    # 3. Lag sentence embeddings ved å ta gjennomsnittet av token embeddings for hver setning
    ref_sents = make_sentence_embeddings(model_input=ref_tokens, model_output=references)
    hyp_sents = make_sentence_embeddings(model_input=hyp_tokens, model_output=hypotheses)

    print("sent_embeds", ref_sents, hyp_sents)
    # 4. Regn ut cosinusdistansen mellom setningsembeddingene
    semdist = [cosine_dist(ref, hyp) for ref, hyp in zip(ref_sents, hyp_sents)]
    print("result", semdist)
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
