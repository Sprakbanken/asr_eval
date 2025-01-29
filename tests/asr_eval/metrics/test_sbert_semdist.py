import pytest
from asr_eval.metrics import sbert_semdist


@pytest.mark.parametrize(
    "reference,hypothesis",
    [
        ("foo foo foo", "foo foo foo"),
        ("Setninger kan være korte.", "Setninger kan være korte."),
        ("Noen er lange.", "Noen er lange."),
        ("Ord", "ord"),
    ],
)
def test_returns_0_for_equal_strings(reference, hypothesis, sbert_model):
    result = sbert_semdist(reference=reference, hypothesis=hypothesis, model=sbert_model)
    assert result == pytest.approx(0, rel=1e-3)
