# %%
import pandas as pd

import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM


# %%
# bertmodelname: str = "ltg/norbert3-large"
# bertmodelname: str = "NbAiLab/nb-bert-large"
bertmodelname: str = "NbAiLab/nb-bert-base"

tokenizer = AutoTokenizer.from_pretrained(bertmodelname)
model = AutoModelForMaskedLM.from_pretrained(bertmodelname, trust_remote_code=True)


# %%
df = pd.read_csv(
    "/home/ingeridd/prosjekter/asr_eval/data/output/2024/2024-12-23_facebook_mms-1b-all.csv"
)

df = df.head(3)

goldcol = "raw_text"
predcol = "predictions"


# %%
# Tokeniser gullstandard og predikerte tekster

ref_tokens = tokenizer(
    df[goldcol].tolist(),
    return_tensors="pt",
    padding=True,
)
hyp_tokens = tokenizer(
    df[predcol].tolist(),
    return_tensors="pt",
    padding=True,
)

# %%
# Send tokeniserte tekster gjennom modellen for å få tokenembeddings

with torch.no_grad():
    reference = model(**ref_tokens)
    hypotheses = model(**hyp_tokens)


# %%
# TODO: Regne gjennomsnittet av parvis sammenligning
# meanpooling


# %%
# Lag sentence embeddings ved å ta gjennomsnittet av token embeddings for hver setning
def make_sentence_embeddings(model_output, model_input):
    embeddings = []
    for token_embeddings, attention_mask in zip(
        model_output.logits, model_input["attention_mask"]
    ):
        if attention_mask[-1] == 1:
            idx = len(token_embeddings) - 1
        else:
            idx = attention_mask.tolist().index(0)
        sentence_embedding = token_embeddings[:idx].mean(axis=0)
        embeddings += [sentence_embedding]
    return embeddings


# %%
ref_sents = make_sentence_embeddings(reference, ref_tokens)

hyp_sents = make_sentence_embeddings(hypotheses, hyp_tokens)


# %%
def cosine_dist(sent1: torch.tensor, sent2: torch.tensor):
    cossim = torch.nn.CosineSimilarity(dim=0)
    return float(1 - cossim(sent1, sent2))


# %%

semdist = [cosine_dist(ref, hyp) for ref, hyp in zip(ref_sents, hyp_sents)]


# %%
float(semdist[0])


# %%
df["semdist"] = semdist
df


# %%

import numpy as np
from dtw import *

alignment = dtw(
    hypotheses.logits[0], reference.logits[0], keep_internals=True, dist_method="cosine"
)

# %%
alignment.plot()

# %%
hypotheses.logits[0].shape

# %%
alignment.normalizedDistance

# %%
alignment_2 = dtw(
    hypotheses.logits[1], reference.logits[1], keep_internals=True, dist_method="cosine"
)

# %%
alignment_2.plot()

# %%
alignment_2.normalizedDistance

# %%
aligned_semdist = []

for sent_pair_index, (referanse_setning, hypotese_setning) in enumerate(
    zip(ref_tokens.input_ids, hyp_tokens.input_ids)
):
    # TODO: regn ut "gjennomsnittlig"
    # semantisk avstand mellom hver mulige phi og velg den minste
    # lag phi-matrisen, regn ut D(phi) for hver phi (og gang med 1/N)
    if tokenizer.pad_token_id in referanse_setning:
        N = list(referanse_setning).index(tokenizer.pad_token_id)
    else:
        N = len(referanse_setning)
    if tokenizer.pad_token_id in hypotese_setning:
        M = list(hypotese_setning).index(tokenizer.pad_token_id)
    else:
        M = len(hypotese_setning)

    reference_embeddings = reference.logits[sent_pair_index]
    hypothesis_embeddings = hypotheses.logits[sent_pair_index]
    print(N)

    # print(referanse_setning[:N])
    # print(reference_embeddings[:N])

    print(M)
    # print(hypotese_setning[:M])
    # print(hypotese_setning[:M])
    alignment = dtw(
        hypothesis_embeddings[:M],
        reference_embeddings[:N],
        keep_internals=True,
        dist_method="cosine",
    )
    print(alignment.normalizedDistance)
    aligned_semdist += [alignment.normalizedDistance]

# %%
df["aligned_semdist"] = aligned_semdist

# %%
df

# %%


# %%


# %%
