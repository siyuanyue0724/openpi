"""Tensor-free validation for fixed-observation distributed training evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

FIXED_OBSERVATION_TRAINING_PAIR_FINGERPRINT_SCHEMA = (
    "picf-next.fixed-observation-training-pair-fingerprint.v1"
)
FIXED_OBSERVATION_TRAINING_PAIR_FINGERPRINT_FIELDS = frozenset(
    {
        "batch_size",
        "controls_sha256",
        "language_masks_sha256",
        "language_tokens_sha256",
        "modalities_sha256",
        "non_language_model_inputs_sha256",
        "routing_source_sha256",
        "schema",
        "structural_source_sha256",
        "task_keys",
    }
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_fixed_observation_training_pair_fingerprints(
    values: Sequence[object],
    *,
    expected_world_size: int = 2,
) -> None:
    """Prove exact non-language equality and distinct language across fixed-X ranks."""

    if (
        isinstance(expected_world_size, bool)
        or not isinstance(expected_world_size, int)
        or expected_world_size != 2
    ):
        raise ValueError("fixed-X training fingerprint validation requires two ranks")
    if (
        not isinstance(values, Sequence)
        or isinstance(values, str | bytes)
        or len(values) != expected_world_size
    ):
        raise ValueError("fixed-X training fingerprints must contain exactly two ranks")
    parsed: list[Mapping[str, object]] = []
    for rank, value in enumerate(values):
        if (
            not isinstance(value, Mapping)
            or set(value) != FIXED_OBSERVATION_TRAINING_PAIR_FINGERPRINT_FIELDS
            or value.get("schema") != FIXED_OBSERVATION_TRAINING_PAIR_FINGERPRINT_SCHEMA
            or value.get("batch_size") != 1
        ):
            raise ValueError(f"fixed-X training fingerprint from rank {rank} is malformed")
        for name in FIXED_OBSERVATION_TRAINING_PAIR_FINGERPRINT_FIELDS - {
            "batch_size",
            "modalities_sha256",
            "schema",
            "task_keys",
        }:
            if not _is_sha256(value.get(name)):
                raise ValueError(
                    f"fixed-X training fingerprint {name!r} from rank {rank} is invalid"
                )
        modality_digest = value.get("modalities_sha256")
        if modality_digest is not None and not _is_sha256(modality_digest):
            raise ValueError(f"fixed-X training modality fingerprint from rank {rank} is invalid")
        task_keys = value.get("task_keys")
        if (
            not isinstance(task_keys, list)
            or len(task_keys) != 1
            or not isinstance(task_keys[0], str)
            or not task_keys[0]
        ):
            raise ValueError(f"fixed-X training task key from rank {rank} is malformed")
        parsed.append(value)

    first, second = parsed
    equal_fields = FIXED_OBSERVATION_TRAINING_PAIR_FINGERPRINT_FIELDS - {
        "language_masks_sha256",
        "language_tokens_sha256",
        "task_keys",
    }
    changed = tuple(name for name in sorted(equal_fields) if first[name] != second[name])
    if changed:
        raise ValueError(f"fixed-X training ranks changed non-language contracts: {changed!r}")
    if first["task_keys"] == second["task_keys"]:
        raise ValueError("fixed-X training ranks retained the same loss-side task")
    if first["language_tokens_sha256"] == second["language_tokens_sha256"]:
        raise ValueError("fixed-X training ranks retained the same tokenized language")


def validate_fixed_observation_training_rank_metadata(
    values: Sequence[object],
    *,
    expected_world_size: int = 2,
) -> bool:
    """Validate all-rank activation and fingerprints from one shared step."""

    if (
        isinstance(expected_world_size, bool)
        or not isinstance(expected_world_size, int)
        or expected_world_size != 2
    ):
        raise ValueError("fixed-X training metadata validation requires two ranks")
    if (
        not isinstance(values, Sequence)
        or isinstance(values, str | bytes)
        or len(values) != expected_world_size
    ):
        raise ValueError("fixed-X training metadata must contain every expected rank")
    digests: list[str | None] = []
    fingerprints: list[object] = []
    required = {"fixed_observation_pair_sha256", "fixed_observation_fingerprint"}
    for rank, value in enumerate(values):
        if not isinstance(value, Mapping) or not required <= set(value):
            raise ValueError(f"fixed-X training metadata from rank {rank} is malformed")
        digest = value["fixed_observation_pair_sha256"]
        if digest is not None and not _is_sha256(digest):
            raise ValueError(f"fixed-X training plan digest from rank {rank} is invalid")
        digests.append(digest)
        fingerprints.append(value["fixed_observation_fingerprint"])
    if len(set(digests)) != 1:
        raise ValueError("fixed-X training activation differs across ranks")
    if digests[0] is None:
        if any(value is not None for value in fingerprints):
            raise ValueError("causal training step unexpectedly carried fixed-X evidence")
        return False
    validate_fixed_observation_training_pair_fingerprints(
        fingerprints,
        expected_world_size=expected_world_size,
    )
    return True
