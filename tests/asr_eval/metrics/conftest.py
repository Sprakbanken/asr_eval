import pytest
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, AutoModelForMaskedLM


@pytest.fixture(scope="session")
def sbert_model():
    model = SentenceTransformer("NbAiLab/nb-sbert-base") 
    return model


@pytest.fixture(scope="session")
def bert_model():
    """Load a "small" Norwegian BERT model to run with the tests, which can also be run on CPU"""
    bertmodelname = "NbAiLab/nb-bert-base"
    model = AutoModelForMaskedLM.from_pretrained(bertmodelname, trust_remote_code=True)
    return model


@pytest.fixture(scope="session")
def bert_tokenizer():
    """Load a BERT tokenizer"""
    bertmodelname = "NbAiLab/nb-bert-base"
    tokenizer = AutoTokenizer.from_pretrained(bertmodelname)
    return tokenizer
