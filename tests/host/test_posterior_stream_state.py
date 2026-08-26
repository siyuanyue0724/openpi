from __future__ import annotations

import copy
from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from picf_next.models.temporal import ObjectBeliefBatch, TemporalFilterConfig  # noqa: E402
from picf_next.training.control import (  # noqa: E402
    EpisodeSampleSequence,
    FrozenEpisodeStreamPlan,
)
from picf_next.training.stream_state import (  # noqa: E402
    PosteriorStreamState,
    PosteriorStreamStateGroup,
)
from tests.geometry_contract import synthetic_geometry_contract  # noqa: E402

GEOMETRY = synthetic_geometry_contract(2)


def _config() -> TemporalFilterConfig:
    return TemporalFilterConfig(
        address_dim=2,
        content_dim=3,
        geometry_dim=2,
        geometry_contract=GEOMETRY,
        action_dim=4,
        reference_delta_t_s=0.1,
        hidden_dim=12,
        num_layers=1,
        num_heads=3,
    )


def _stream(
    *,
    max_parameter_lag: int = 1,
    dtype: torch.dtype = torch.float32,
) -> PosteriorStreamState:
    return PosteriorStreamState(
        _config(),
        lane_ids=("rank-0/lane-0", "rank-0/lane-1"),
        capacity=3,
        dtype=dtype,
        max_parameter_lag=max_parameter_lag,
    )


def _plan() -> FrozenEpisodeStreamPlan:
    episodes = tuple(
        EpisodeSampleSequence(
            f"episode-{episode_index}",
            tuple(
                f"episode-{episode_index}/frame-{transition_index}"
                for transition_index in range(3 + episode_index)
            ),
        )
        for episode_index in range(4)
    )
    return FrozenEpisodeStreamPlan(
        dataset_id="stream-state-fixture",
        dataset_revision="v1",
        dataset_manifest_sha256="a" * 64,
        episodes=episodes,
        comparison_id="stream-state-seed-47",
        seed=47,
        global_batch_size=4,
        total_steps=20,
    )


def _populated_belief(stream: PosteriorStreamState) -> ObjectBeliefBatch:
    valid = torch.tensor([[True, False, False], [True, True, False]])

    def floating_rows(width: int, offset: float) -> torch.Tensor:
        values = torch.arange(2 * 3 * width, dtype=torch.float32).reshape(2, 3, width)
        return (values + offset) * valid.unsqueeze(-1)

    address = torch.nn.functional.normalize(floating_rows(2, 0.1), dim=-1)
    address = (address * valid.unsqueeze(-1)).requires_grad_()
    return ObjectBeliefBatch(
        address_mean=address,
        content_mean=floating_rows(3, 0.2).requires_grad_(),
        geometry_mean=floating_rows(2, 0.3).requires_grad_(),
        geometry_covariance_diag=torch.full((2, 3, 2), 0.2) * valid.unsqueeze(-1),
        existence_logits=torch.full((2, 3), 2.0) * valid,
        visibility_given_existence_logits=torch.full((2, 3), 1.0) * valid,
        measurement_age_s=torch.tensor([[0.4, 0.0, 0.0], [0.2, 0.7, 0.0]]) * valid,
        valid=valid,
        age=torch.tensor([[4, 0, 0], [2, 7, 0]], dtype=torch.long),
    )


def _assert_belief_equal(left: ObjectBeliefBatch, right: ObjectBeliefBatch) -> None:
    for field in left.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(left, field),
            getattr(right, field),
            rtol=0.0,
            atol=0.0,
        )


def test_bfloat16_stream_accepts_and_serializes_representable_unit_addresses() -> None:
    stream = _stream(dtype=torch.bfloat16)
    initial = stream.prepare_chunk(
        episode_keys=("episode-a", "episode-b"),
        start_transition_indices=(0, 0),
        current_parameter_version=0,
    )
    valid = torch.tensor([[True, False, False], [True, True, False]])
    address = torch.tensor(
        [
            [[0.8, 0.6], [0.0, 0.0], [0.0, 0.0]],
            [[0.6, 0.8], [2**-0.5, 2**-0.5], [0.0, 0.0]],
        ],
        dtype=torch.bfloat16,
    )
    belief = ObjectBeliefBatch(
        address_mean=address,
        content_mean=torch.zeros_like(initial.content_mean),
        geometry_mean=torch.zeros_like(initial.geometry_mean),
        geometry_covariance_diag=(
            torch.full_like(initial.geometry_covariance_diag, 0.25) * valid.unsqueeze(-1)
        ),
        existence_logits=torch.ones_like(initial.existence_logits) * valid,
        visibility_given_existence_logits=torch.ones_like(initial.existence_logits) * valid,
        measurement_age_s=torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.2, 0.0]], dtype=torch.bfloat16),
        valid=valid,
        age=torch.tensor([[1, 0, 0], [1, 2, 0]], dtype=torch.long),
    )

    assert stream.candidate_value_validity(belief)
    stream.commit_chunk(
        belief,
        transition_count=1,
        state_parameter_version=0,
    )
    state = stream.state_dict()
    restored = _stream(dtype=torch.bfloat16)
    restored.load_state_dict(state)
    _assert_belief_equal(restored.belief, belief)


def _assert_nested_state_equal(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_state_equal(left[key], right[key])
    elif isinstance(left, list):
        assert isinstance(right, list)
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right, strict=True):
            _assert_nested_state_equal(left_value, right_value)
    else:
        assert left == right


def test_stream_carries_detached_posterior_without_replaying_history() -> None:
    stream = _stream()
    initial = stream.prepare_chunk(
        episode_keys=("episode-a", "episode-b"),
        start_transition_indices=(0, 0),
        current_parameter_version=0,
    )
    assert not initial.valid.any()

    final = _populated_belief(stream)
    stream.commit_chunk(final, transition_count=4, state_parameter_version=0)
    assert stream.next_transition_indices == (4, 4)
    assert all(
        not getattr(stream.belief, field).requires_grad
        for field in stream.belief.__dataclass_fields__
    )

    carried = stream.prepare_chunk(
        episode_keys=("episode-a", "episode-b"),
        start_transition_indices=(4, 4),
        current_parameter_version=1,
    )
    _assert_belief_equal(carried, final)
    assert carried.address_mean.data_ptr() != stream.belief.address_mean.data_ptr()
    stream.abort_chunk()

    with pytest.raises(ValueError, match="parameter-version lag"):
        stream.prepare_chunk(
            episode_keys=("episode-a", "episode-b"),
            start_transition_indices=(4, 4),
            current_parameter_version=2,
        )


def test_episode_replacement_resets_only_the_reassigned_lane() -> None:
    stream = _stream()
    stream.prepare_chunk(
        episode_keys=("episode-a", "episode-b"),
        start_transition_indices=(0, 0),
        current_parameter_version=3,
    )
    final = _populated_belief(stream)
    stream.commit_chunk(final, transition_count=2, state_parameter_version=3)

    initial = stream.prepare_chunk(
        episode_keys=("episode-c", "episode-b"),
        start_transition_indices=(0, 2),
        current_parameter_version=4,
    )
    for field in initial.__dataclass_fields__:
        actual = getattr(initial, field)
        expected = getattr(final, field)
        assert not actual[0].any()
        torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)


def test_stream_rejects_discontinuity_and_nonzero_episode_birth() -> None:
    stream = _stream()
    stream.prepare_chunk(
        episode_keys=("episode-a", "episode-b"),
        start_transition_indices=(0, 0),
        current_parameter_version=0,
    )
    stream.commit_chunk(_populated_belief(stream), transition_count=3, state_parameter_version=0)

    with pytest.raises(ValueError, match="discontinuous"):
        stream.prepare_chunk(
            episode_keys=("episode-a", "episode-b"),
            start_transition_indices=(2, 3),
            current_parameter_version=1,
        )
    with pytest.raises(ValueError, match="transition zero"):
        stream.prepare_chunk(
            episode_keys=("episode-c", "episode-b"),
            start_transition_indices=(1, 3),
            current_parameter_version=1,
        )


def test_stream_rejects_future_state_and_uncommitted_operations() -> None:
    stream = _stream(max_parameter_lag=4)
    with pytest.raises(RuntimeError, match="prepare_chunk"):
        stream.commit_chunk(
            _populated_belief(stream),
            transition_count=1,
            state_parameter_version=0,
        )
    stream.prepare_chunk(
        episode_keys=("episode-a", "episode-b"),
        start_transition_indices=(0, 0),
        current_parameter_version=3,
    )
    with pytest.raises(RuntimeError, match="already pending"):
        stream.prepare_chunk(
            episode_keys=("episode-a", "episode-b"),
            start_transition_indices=(0, 0),
            current_parameter_version=3,
        )
    with pytest.raises(RuntimeError, match="uncommitted"):
        stream.state_dict()
    stream.commit_chunk(_populated_belief(stream), transition_count=1, state_parameter_version=3)

    with pytest.raises(ValueError, match="future parameter version"):
        stream.prepare_chunk(
            episode_keys=("episode-a", "episode-b"),
            start_transition_indices=(1, 1),
            current_parameter_version=2,
        )


@pytest.mark.parametrize("invalid_version", [True, -1, 1])
def test_commit_requires_the_exact_prepared_parameter_version(invalid_version: object) -> None:
    stream = _stream()
    stream.prepare_chunk(
        episode_keys=("episode-a", "episode-b"),
        start_transition_indices=(0, 0),
        current_parameter_version=0,
    )
    with pytest.raises(ValueError, match="state_parameter_version"):
        stream.commit_chunk(
            _populated_belief(stream),
            transition_count=1,
            state_parameter_version=invalid_version,  # type: ignore[arg-type]
        )
    stream.abort_chunk()


def test_stream_checkpoint_roundtrip_restores_exact_cursor_and_belief() -> None:
    stream = _stream(max_parameter_lag=2)
    stream.prepare_chunk(
        episode_keys=("episode-a", "episode-b"),
        start_transition_indices=(0, 0),
        current_parameter_version=7,
    )
    expected = _populated_belief(stream)
    expected_tracks = (("track:a", None, None), ("track:b", "track:c", None))
    stream.commit_chunk(
        expected,
        transition_count=5,
        state_parameter_version=7,
        final_loss_track_keys_by_row=expected_tracks,
    )
    payload = stream.state_dict()

    restored = _stream(max_parameter_lag=2)
    restored.load_state_dict(payload)
    assert restored.episode_keys == ("episode-a", "episode-b")
    assert restored.next_transition_indices == (5, 5)
    assert restored.state_parameter_versions == (7, 7)
    assert restored.loss_track_keys_by_row == expected_tracks
    _assert_belief_equal(restored.belief, expected)

    payload["belief"]["address_mean"].add_(1000)
    _assert_belief_equal(restored.belief, expected)
    restored.prepare_chunk(
        episode_keys=("episode-a", "episode-b"),
        start_transition_indices=(5, 5),
        current_parameter_version=9,
    )
    assert restored.pending_loss_track_keys_by_row == expected_tracks


def test_episode_replacement_resets_only_its_loss_track_rows() -> None:
    stream = _stream()
    stream.prepare_chunk(
        episode_keys=("episode-a", "episode-b"),
        start_transition_indices=(0, 0),
        current_parameter_version=0,
    )
    tracks = (("track:a", None, None), ("track:b", "track:c", None))
    stream.commit_chunk(
        _populated_belief(stream),
        transition_count=2,
        state_parameter_version=0,
        final_loss_track_keys_by_row=tracks,
    )

    stream.prepare_chunk(
        episode_keys=("episode-c", "episode-b"),
        start_transition_indices=(0, 2),
        current_parameter_version=1,
    )

    assert stream.pending_loss_track_keys_by_row == (
        (None, None, None),
        tracks[1],
    )


@pytest.mark.parametrize(
    "tracks, message",
    [
        ((("track:a", "track:a", None), ("track:b", None, None)), "unique"),
        ((("track:a", "track:invalid", None), ("track:b", None, None)), "unused"),
        ((("track:a", None), ("track:b", None, None)), "lane-by-posterior-row"),
    ],
)
def test_invalid_loss_tracks_cannot_partially_commit_stream_state(
    tracks: tuple[tuple[str | None, ...], ...],
    message: str,
) -> None:
    stream = _stream()
    before = copy.deepcopy(stream.state_dict())
    stream.prepare_chunk(
        episode_keys=("episode-a", "episode-b"),
        start_transition_indices=(0, 0),
        current_parameter_version=0,
    )

    with pytest.raises(ValueError, match=message):
        stream.commit_chunk(
            _populated_belief(stream),
            transition_count=1,
            state_parameter_version=0,
            final_loss_track_keys_by_row=tracks,
        )

    assert stream.has_pending_chunk
    assert stream.episode_keys == (None, None)
    assert stream.next_transition_indices == (0, 0)
    assert stream.loss_track_keys_by_row == ((None, None, None),) * 2
    stream.abort_chunk()
    _assert_nested_state_equal(stream.state_dict(), before)


def test_stream_checkpoint_rejects_schema_metadata_and_tensor_corruption() -> None:
    stream = _stream()
    payload = stream.state_dict()

    legacy = copy.deepcopy(payload)
    legacy["schema"] = "picf-next.posterior-stream-state.v4"
    legacy.pop("factorization")
    with pytest.raises(ValueError, match="unsupported posterior stream state schema"):
        stream.load_state_dict(legacy)

    wrong_factorization = copy.deepcopy(payload)
    wrong_factorization["factorization"] = "legacy-dynamic-gaussian.v0"
    with pytest.raises(ValueError, match="factorization"):
        stream.load_state_dict(wrong_factorization)

    extra = copy.deepcopy(payload)
    extra["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        stream.load_state_dict(extra)

    wrong_lanes = copy.deepcopy(payload)
    wrong_lanes["lane_ids"] = ["rank-1/lane-0", "rank-1/lane-1"]
    with pytest.raises(ValueError, match="lane identities"):
        stream.load_state_dict(wrong_lanes)

    inconsistent = copy.deepcopy(payload)
    inconsistent["episode_keys"][0] = "episode-a"
    with pytest.raises(ValueError, match="internally inconsistent"):
        stream.load_state_dict(inconsistent)

    nonfinite = copy.deepcopy(payload)
    nonfinite["belief"]["address_mean"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        stream.load_state_dict(nonfinite)

    wrong_dtype = copy.deepcopy(payload)
    wrong_dtype["belief"]["age"] = wrong_dtype["belief"]["age"].to(torch.int32)
    with pytest.raises(ValueError, match="dtype"):
        stream.load_state_dict(wrong_dtype)

    invalid_row = copy.deepcopy(payload)
    invalid_row["belief"]["address_mean"][0, 0, 0] = 1.0
    with pytest.raises(ValueError, match="invalid stream belief rows"):
        stream.load_state_dict(invalid_row)

    invalid_measurement_age = copy.deepcopy(payload)
    invalid_measurement_age["belief"]["measurement_age_s"][0, 0] = -0.1
    with pytest.raises(ValueError, match="measurement age"):
        stream.load_state_dict(invalid_measurement_age)

    before = copy.deepcopy(stream.state_dict())
    stream.validate_state_dict(payload)
    _assert_nested_state_equal(stream.state_dict(), before)


def test_device_side_candidate_validation_enforces_posterior_row_invariants() -> None:
    stream = _stream()
    valid = _populated_belief(stream)
    assert bool(stream.candidate_value_validity(valid))

    invalid_padding = _populated_belief(stream)
    with torch.no_grad():
        invalid_padding.content_mean[0, 1, 0] = 1.0
    assert not bool(stream.candidate_value_validity(invalid_padding))

    low_covariance = _populated_belief(stream)
    low_covariance.geometry_covariance_diag[0, 0, 0] = _config().minimum_variance / 2
    assert not bool(stream.candidate_value_validity(low_covariance))

    invalid_measurement_age = _populated_belief(stream)
    invalid_measurement_age.measurement_age_s[0, 0] = -0.1
    assert not bool(stream.candidate_value_validity(invalid_measurement_age))


def test_rank_group_consumes_planned_single_transitions_and_roundtrips() -> None:
    plan = _plan()
    group = PosteriorStreamStateGroup.for_rank_partition(
        _config(),
        plan,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=2,
        capacity=3,
        dtype=torch.float32,
    )
    assert group.stream_names == ("accumulation-00000", "accumulation-00001")
    expected = {}
    for accumulation_index, name in enumerate(group.stream_names):
        microbatch = plan.microbatch_for_rank(
            0,
            rank=0,
            world_size=1,
            gradient_accumulation_steps=2,
            accumulation_index=accumulation_index,
        )
        stream = group[name]
        initial = stream.prepare_planned_transitions(
            microbatch.transitions,
            current_parameter_version=0,
        )
        assert not initial.valid.any()
        final = _populated_belief(stream)
        stream.commit_chunk(final, transition_count=1, state_parameter_version=0)
        expected[name] = final

    restored = PosteriorStreamStateGroup.for_rank_partition(
        _config(),
        plan,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=2,
        capacity=3,
        dtype=torch.float32,
    )
    restored.load_state_dict(group.state_dict())
    for accumulation_index, name in enumerate(restored.stream_names):
        microbatch = plan.microbatch_for_rank(
            1,
            rank=0,
            world_size=1,
            gradient_accumulation_steps=2,
            accumulation_index=accumulation_index,
        )
        carried = restored[name].prepare_planned_transitions(
            microbatch.transitions,
            current_parameter_version=1,
        )
        _assert_belief_equal(carried, expected[name])
        restored[name].abort_chunk()


def test_planned_transition_lane_order_fails_closed() -> None:
    plan = _plan()
    group = PosteriorStreamStateGroup.for_rank_partition(
        _config(),
        plan,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=2,
        capacity=3,
    )
    microbatch = plan.microbatch_for_rank(
        0,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=2,
        accumulation_index=0,
    )
    with pytest.raises(ValueError, match="lanes differ"):
        group["accumulation-00000"].prepare_planned_transitions(
            tuple(reversed(microbatch.transitions)),
            current_parameter_version=0,
        )


def test_group_load_is_atomic_when_one_stream_is_corrupt() -> None:
    plan = _plan()
    source = PosteriorStreamStateGroup.for_rank_partition(
        _config(),
        plan,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=2,
        capacity=3,
    )
    for accumulation_index, name in enumerate(source.stream_names):
        microbatch = plan.microbatch_for_rank(
            0,
            rank=0,
            world_size=1,
            gradient_accumulation_steps=2,
            accumulation_index=accumulation_index,
        )
        stream = source[name]
        stream.prepare_planned_transitions(
            microbatch.transitions,
            current_parameter_version=0,
        )
        stream.commit_chunk(
            _populated_belief(stream),
            transition_count=1,
            state_parameter_version=0,
        )
    payload = source.state_dict()
    payload["streams"]["accumulation-00001"]["schema"] = "corrupt"

    target = PosteriorStreamStateGroup.for_rank_partition(
        _config(),
        plan,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=2,
        capacity=3,
    )
    with pytest.raises(ValueError, match="schema"):
        target.load_state_dict(payload)
    assert all(not target[name].belief.valid.any() for name in target.stream_names)


def test_group_commit_validates_every_shard_before_mutating_any_cursor() -> None:
    plan = _plan()
    group = PosteriorStreamStateGroup.for_rank_partition(
        _config(),
        plan,
        rank=0,
        world_size=1,
        gradient_accumulation_steps=2,
        capacity=3,
    )
    final_beliefs = {}
    for accumulation_index, name in enumerate(group.stream_names):
        microbatch = plan.microbatch_for_rank(
            0,
            rank=0,
            world_size=1,
            gradient_accumulation_steps=2,
            accumulation_index=accumulation_index,
        )
        stream = group[name]
        stream.prepare_planned_transitions(
            microbatch.transitions,
            current_parameter_version=0,
        )
        final_beliefs[name] = _populated_belief(stream)
    invalid = final_beliefs["accumulation-00001"]
    final_beliefs["accumulation-00001"] = replace(
        invalid,
        address_mean=invalid.address_mean[:, :2],
    )

    with pytest.raises(ValueError, match="incompatible shape"):
        group.commit_prepared_chunks(
            final_beliefs,
            transition_count=1,
            state_parameter_version=0,
        )
    assert group.has_pending_chunks
    assert all(group[name].next_transition_indices == (0, 0) for name in group.stream_names)
    assert all(not group[name].belief.valid.any() for name in group.stream_names)
    group.abort_pending_chunks()
    assert not group.has_pending_chunks
