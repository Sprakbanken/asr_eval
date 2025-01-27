import pytest
from asr_eval.metrics import aligned_semdist


@pytest.mark.parametrize(
    "reference,hypothesis",
    [
        ("foo foo foo", "foo foo foo"),
        ("Setninger kan være korte.", "Setninger kan være korte."),
        ("Noen er lange.", "Noen er lange."),
        ("Ord", "Ord"),
    ],
)
def test_returns_0_for_equal_strings(reference, hypothesis, bert_tokenizer, bert_model):
    result = aligned_semdist(reference=reference, hypothesis=hypothesis, model=bert_model, tokenizer=bert_tokenizer)
    assert result.score == 0
