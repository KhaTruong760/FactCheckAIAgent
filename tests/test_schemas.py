import pytest
from pydantic import ValidationError

from schemas import Verdict


def _verdict(label):
    return Verdict(claim_id="c1", label=label, confidence=0.9, quote="q", evidence_id="e1", reason="r")


@pytest.mark.parametrize("label", ["SUPPORTED", "PARTIAL", "UNSUPPORTED", "CONTRADICTED"])
def test_verdict_accepts_valid_labels(label):
    assert _verdict(label).label == label


def test_verdict_rejects_unknown_label():
    with pytest.raises(ValidationError):
        _verdict("UNSUPPORTTED")
