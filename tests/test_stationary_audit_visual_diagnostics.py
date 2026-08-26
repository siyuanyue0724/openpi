from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tools.audit_stationary_temporal_candidate import (  # noqa: E402
    _aligned_probabilities,
    _attach_camera_ownership_masses,
    _format_visual_diagnostics,
    _overlay,
    _publish_visual_manifest,
    _terminal_lifecycle_history,
    _visual_history_score,
)


def test_aligned_visual_diagnostics_expose_confidence_and_current_match_mass() -> None:
    target = SimpleNamespace(
        ownership=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        temporal_identity_keys=("object:a", "object:b"),
    )
    lifecycle = SimpleNamespace(
        alive_identity_keys=("object:a", "object:b", "object:c"),
        visibility=torch.tensor([1.0, 1.0, 0.0]),
        visibility_supervised=torch.tensor([True, True, True]),
    )
    supervision = SimpleNamespace(set_targets=(target,), lifecycle_targets=(lifecycle,))
    match = SimpleNamespace(
        prediction_indices=torch.tensor([1, 0]),
        target_indices=torch.tensor([0, 1]),
    )
    trainer = SimpleNamespace(
        objective=SimpleNamespace(
            set_criterion=lambda discovery, targets: SimpleNamespace(matches=(match,))
        )
    )
    discovery = SimpleNamespace(
        context_ownership=torch.tensor([[0.1, 0.2, 0.7]]),
        ownership=torch.tensor(
            [
                [
                    [0.2, 0.7],
                    [0.8, 0.1],
                    [0.0, 0.2],
                ]
            ]
        ),
        existence=torch.tensor([[0.6, 0.8]]),
        localization_confidence=torch.tensor([[0.5, 0.75]]),
        measurement_probability=torch.tensor([[0.3, 0.6]]),
    )
    belief = SimpleNamespace(
        valid=torch.tensor([[True, True, True]]),
        existence=torch.tensor([[0.9, 0.4, 0.85]]),
        visibility=torch.tensor([[0.45, 0.1, 0.05]]),
    )
    posterior = SimpleNamespace(
        ownership=torch.tensor(
            [
                [
                    [0.4, 0.1, 0.0],
                    [0.2, 0.0, 0.0],
                    [0.0, 0.3, 0.0],
                ]
            ]
        ),
        belief=belief,
        match_probability=torch.tensor([[[0.25, 0.15], [0.01, 0.02], [0.0, 0.0]]]),
        map_present=torch.tensor([[True, False, True]]),
    )
    final = SimpleNamespace(discovery=discovery, posterior=posterior)
    output = SimpleNamespace(
        train_outputs=(final,),
        objective=SimpleNamespace(loss_track_keys_by_row=(("object:a", "object:b", "object:c"),)),
    )

    aligned_target, aligned_discovery, aligned_posterior, diagnostics = _aligned_probabilities(
        trainer,
        output,
        supervision,
    )
    diagnostics = _attach_camera_ownership_masses(
        diagnostics,
        target=aligned_target,
        discovery=aligned_discovery,
        posterior=aligned_posterior,
        image_spans=(
            SimpleNamespace(image_key="observation.images.image", start=0, stop=2),
            SimpleNamespace(image_key="observation.images.wrist_image", start=2, stop=3),
        ),
        vision_start=0,
    )

    np.testing.assert_allclose(aligned_target[:, 2], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(aligned_discovery[:, 0], [0.7, 0.1, 0.2])
    np.testing.assert_allclose(aligned_discovery[:, 1], [0.2, 0.8, 0.0])
    np.testing.assert_allclose(aligned_discovery[:, 2], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(aligned_posterior[:, 0], [0.4, 0.2, 0.0])
    assert diagnostics[0] == {
        "identity_key": "object:a",
        "target_currently_measurable": True,
        "target_visibility": 1.0,
        "target_visibility_supervised": True,
        "target_ownership_mass": 1.0,
        "discovery_existence": 0.800000011920929,
        "discovery_localization": 0.75,
        "discovery_measurement": 0.6000000238418579,
        "discovery_ownership_mass": 1.0,
        "posterior_association": 0.4000000059604645,
        "posterior_existence": 0.8999999761581421,
        "posterior_visibility": 0.44999998807907104,
        "posterior_ownership_mass": 0.6000000238418579,
        "posterior_map_present": True,
        "camera_ownership_mass": {
            "observation.images.image": {
                "target": 1.0,
                "discovery": 0.800000011920929,
                "posterior": 0.6000000238418579,
            },
            "observation.images.wrist_image": {
                "target": 0.0,
                "discovery": 0.20000000298023224,
                "posterior": 0.0,
            },
        },
    }
    assert diagnostics[2]["target_currently_measurable"] is False
    assert diagnostics[2]["target_visibility"] == 0.0
    assert diagnostics[2]["discovery_measurement"] is None
    assert diagnostics[2]["posterior_map_present"] is True
    assert diagnostics[2]["posterior_association"] == 0.0
    assert diagnostics[2]["posterior_visibility"] == pytest.approx(0.05)
    rendered = _format_visual_diagnostics("candidate", diagnostics[0])
    assert "meas=0.600" in rendered
    assert "assoc=0.400" in rendered
    assert "vis=0.450" in rendered
    assert "ext:D=0.80,P=0.60" in rendered
    assert "wrist:D=0.20,P=0.00" in rendered


def test_visual_overlay_uses_probability_mass_instead_of_hard_argmax() -> None:
    source = np.zeros((2, 2, 3), dtype=np.uint8)
    probability = np.asarray(
        [
            [0.01, 0.00, 0.99],
            [1.00, 0.00, 0.00],
            [0.00, 0.00, 1.00],
            [0.00, 0.00, 1.00],
        ],
        dtype=np.float32,
    )

    rendered = _overlay(source, probability)

    assert int(rendered[0, 1].sum()) > 100
    assert int(rendered[0, 0].sum()) < 10
    assert int(rendered[1, 0].sum()) == 0


def test_terminal_history_distinguishes_seen_occlusion_from_never_seen() -> None:
    class Batch:
        observations = (None, None, None, None)
        source_indices_by_frame = ((10,), (11,), (12,), (13,))

        @staticmethod
        def build_supervision(frame_index: int) -> SimpleNamespace:
            measurable = ("object:a",) if frame_index < 2 else ()
            target = SimpleNamespace(temporal_identity_keys=measurable)
            lifecycle = SimpleNamespace(alive_identity_keys=("object:a", "object:b"))
            return SimpleNamespace(set_targets=(target,), lifecycle_targets=(lifecycle,))

    diagnostics = (
        {
            "identity_key": "object:a",
            "target_currently_measurable": False,
            "posterior_existence": 0.2,
            "posterior_map_present": False,
        },
        {
            "identity_key": "object:b",
            "target_currently_measurable": False,
            "posterior_existence": None,
            "posterior_map_present": False,
        },
    )

    history = _terminal_lifecycle_history(Batch(), diagnostics)

    assert history["object:a"] == {
        "ever_measurable_before_final": True,
        "last_measurable_global_index": 11,
        "terminal_unmeasurable_frames": 2,
        "seen_then_unmeasurable": True,
        "candidate_posterior_identity_retained": True,
        "candidate_posterior_map_present": False,
        "candidate_posterior_existence": 0.2,
    }
    assert history["object:b"]["ever_measurable_before_final"] is False
    assert history["object:b"]["seen_then_unmeasurable"] is False
    assert history["object:b"]["last_measurable_global_index"] is None
    assert history["object:b"]["candidate_posterior_identity_retained"] is False


def test_visual_selection_prefers_the_longest_seen_occlusion() -> None:
    never_seen = {"lifecycle_targets": []}
    retained_short = {
        "lifecycle_targets": [
            {
                "seen_then_unmeasurable": True,
                "terminal_unmeasurable_frames": 8,
                "candidate_posterior_identity_retained": True,
            }
        ]
    }
    lost_long = {
        "lifecycle_targets": [
            {
                "seen_then_unmeasurable": True,
                "terminal_unmeasurable_frames": 24,
                "candidate_posterior_identity_retained": False,
            }
        ]
    }

    assert _visual_history_score(never_seen) == (0, 0, 0)
    assert _visual_history_score(retained_short) == (1, 8, 1)
    assert _visual_history_score(lost_long) == (1, 24, 0)
    assert _visual_history_score(lost_long) > _visual_history_score(retained_short)


def test_invalid_visual_manifest_is_never_published(tmp_path) -> None:
    with pytest.raises(ValueError, match="fields differ from its frozen schema"):
        _publish_visual_manifest(tmp_path, {})

    assert not (tmp_path / "visual_artifacts.json").exists()
