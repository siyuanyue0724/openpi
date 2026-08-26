from __future__ import annotations

import pytest

from adr141.tools.audit_training_windows import _segment_id


@pytest.mark.parametrize(
    ("sample_key", "expected"),
    (
        (
            "calvin-language-segment-00000123/transition-00000456-frame-00000789",
            123,
        ),
        ("calvin-source-episode-00000023/frame-00103795", 23),
    ),
)
def test_segment_id_accepts_registered_calvin_sample_key_schemas(
    sample_key: str,
    expected: int,
) -> None:
    assert _segment_id({"sample_keys": [sample_key]}) == expected


def test_segment_id_rejects_unregistered_sample_key_schema() -> None:
    with pytest.raises(ValueError, match="malformed CALVIN sample key"):
        _segment_id({"sample_keys": ["calvin-episode-23/frame-103795"]})
