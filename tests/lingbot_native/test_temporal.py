from __future__ import annotations

import inspect
import json
import random

import pytest
import torch

from picf_next.lingbot_native.addresses import EpisodeAddressState
from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.host import (
    LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
    LingBotNativeGraph,
    LingBotNativeGraphConfig,
    LingBotNativePriorStepper,
)
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.lingbot_native.state import (
    AddressedLayerwisePosteriorState,
    NativeLayerwisePosteriorState,
    NativePosteriorState,
    NativeVidEoMTPairedPosteriorState,
)
from picf_next.lingbot_native.temporal import (
    FROZEN_AUXILIARY_SAMPLING,
    NativeLaneConfig,
    NativeLaneError,
    NativeLaneStamp,
    NativeTrainingLaneBank,
    TemporalBatchPlan,
    TemporalCostProfile,
    TemporalEstimatorConfig,
    TemporalWorkload,
    native_temporal_batch_seed,
    rollout_native_prior_prediction,
    sample_temporal_batch_plan,
)
from picf_next.training.control import EpisodeSampleSequence, FrozenEpisodeStreamPlan


def _estimator(**overrides: object) -> TemporalEstimatorConfig:
    values = {
        "local_bptt_probability": 0.2,
        "overshoot_probability": 0.1,
        "source_mask_probability": 0.1,
        "maximum_optimizer_lag": 8,
    }
    values.update(overrides)
    return TemporalEstimatorConfig(**values)


def _lane_bank() -> NativeTrainingLaneBank:
    return NativeTrainingLaneBank(
        NativeLaneConfig(
            model_digest="lingbot-native-test",
            schema_digest="rows-v1",
            capacity=2,
            host_width=4,
            maximum_optimizer_lag=8,
        )
    )


def _state(value: float) -> NativePosteriorState:
    return NativePosteriorState(torch.full((1, 2, 4), value, dtype=torch.float32))


def _layerwise_state(value: float) -> NativeLayerwisePosteriorState:
    return NativeLayerwisePosteriorState(torch.full((1, 3, 2, 4), value, dtype=torch.float32))


def _addressed_layerwise_state(
    value: float,
    *,
    codebook_sha256: str = "a" * 64,
    architecture_identity: str = LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
) -> AddressedLayerwisePosteriorState:
    return AddressedLayerwisePosteriorState(
        layer_rows=torch.full((1, 3, 2, 4), value, dtype=torch.float32),
        episode_address_state=EpisodeAddressState(
            permutation=torch.tensor([[1, 0]], dtype=torch.long),
            codebook_sha256=codebook_sha256,
        ),
        architecture_identity=architecture_identity,
    )


def _paired_videomt_state(value: float) -> NativeVidEoMTPairedPosteriorState:
    return NativeVidEoMTPairedPosteriorState(
        layer_rows=torch.full((1, 3, 2, 4), value, dtype=torch.bfloat16),
        source_queries=torch.full((1, 2, 6), value + 10, dtype=torch.float32),
        architecture_identity="native-videomt-query-posterior/v1",
    )


def _control(value: float) -> ExecutedControlBatch:
    return ExecutedControlBatch(
        values=torch.full((1, 1, 2), value),
        field_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        token_valid=torch.ones(1, 1, dtype=torch.bool),
        delta_time=torch.full((1, 1), 0.1),
        reset=torch.zeros(1, 1, dtype=torch.bool),
        acknowledged=torch.ones(1, 1, dtype=torch.bool),
    )


@pytest.mark.parametrize("factor", (4, 8))
def test_interleaved_stream_drives_exact_lagged_lane_state_and_resume(
    factor: int,
) -> None:
    episodes = tuple(
        EpisodeSampleSequence(
            f"episode-{episode_index:02d}",
            tuple(
                f"episode-{episode_index:02d}/frame-{frame_index}"
                for frame_index in range(2 + episode_index % 3)
            ),
        )
        for episode_index in range(12)
    )
    plan = FrozenEpisodeStreamPlan(
        dataset_id="interleaved-lane-state-test",
        dataset_revision="v1",
        dataset_manifest_sha256="a" * 64,
        episodes=episodes,
        comparison_id="interleaved-lane-state-test",
        seed=97,
        global_batch_size=1,
        total_steps=factor * 10,
        lane_interleave_factor=factor,
    )
    config = NativeLaneConfig(
        model_digest="interleaved-lane-model",
        schema_digest=plan.plan_sha256,
        capacity=2,
        host_width=4,
        maximum_optimizer_lag=factor,
    )
    bank = NativeTrainingLaneBank(config)
    saw_continuation = False
    saw_reset_replacement = False
    seen_lanes: set[int] = set()

    for optimizer_step in range(plan.total_steps):
        (transition,) = plan.global_batch(optimizer_step).transitions
        lane_id = plan.lane_ids.index(transition.lane_id)
        reset = transition.transition_index == 0
        if lane_id in seen_lanes and reset:
            saw_reset_replacement = True
        if not reset:
            read = bank.read(
                lane_id,
                episode_key=transition.episode_instance_id,
                next_frame_index=transition.transition_index,
                optimizer_step=optimizer_step,
                source_weight_version=3,
            )
            assert read is not None
            assert read.optimizer_lag == factor
            assert read.stamp.frame_index + 1 == transition.transition_index
            saw_continuation = True
        transaction = bank.stage(
            lane_id,
            _state(float(optimizer_step + 1)),
            NativeLaneStamp(
                episode_key=transition.episode_instance_id,
                frame_index=transition.transition_index,
                state_age=transition.transition_index,
                producer_optimizer_step=optimizer_step,
                source_weight_version=3,
            ),
            reset=reset,
        )
        bank.commit_after_optimizer(
            transaction,
            successful_optimizer_step=optimizer_step + 1,
        )
        seen_lanes.add(lane_id)
        if optimizer_step == factor * 5 - 1:
            bank = NativeTrainingLaneBank.deserialize(config, bank.serialize())

    assert seen_lanes == set(range(factor))
    assert saw_continuation
    assert saw_reset_replacement


class _ToyOfficialHost(torch.nn.Module):
    """Exercises the exact graph hook used by the official joint host."""

    def __init__(self, graph: LingBotNativeGraph) -> None:
        super().__init__()
        self.picf_native_graph = graph

    def forward(self, **kwargs: object) -> tuple[list[torch.Tensor | None], None, list]:
        prepared, mask, _, _, runtime = self.picf_native_graph.prepare_joint_inputs(
            inputs_embeds=kwargs["inputs_embeds"],
            attention_mask=kwargs["attention_mask"],
            position_ids=kwargs["position_ids"],
            visual_pos_masks=kwargs["visual_pos_masks"],
            context=kwargs["picf_native_context"],
        )
        assert prepared[0] is not None and prepared[1] is None
        hidden = prepared[0]
        for _ in range(self.picf_native_graph.config.num_layers):
            weight = mask.to(hidden.dtype)
            weight = weight / weight.sum(dim=-1, keepdim=True).clamp_min(1)
            hidden = torch.nn.functional.layer_norm(hidden + weight @ hidden, (hidden.shape[-1],))
        outputs: list[torch.Tensor | None] = [hidden, None]
        self.picf_native_graph.finalize_joint_outputs(outputs_embeds=outputs, runtime=runtime)
        return outputs, None, []


class _ToyOfficialPolicy(torch.nn.Module):
    """Exposes the dedicated root method used by sharded production rollout."""

    def __init__(self, graph: LingBotNativeGraph) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.qwenvl_with_expert = _ToyOfficialHost(graph)

    def picf_native_prior_forward(self, **kwargs: object):
        return self.model.qwenvl_with_expert.forward(**kwargs)


def _stamp(
    frame: int,
    *,
    age: int,
    optimizer: int,
    episode: str = "episode-a",
    source_version: int = 3,
) -> NativeLaneStamp:
    return NativeLaneStamp(
        episode_key=episode,
        frame_index=frame,
        state_age=age,
        producer_optimizer_step=optimizer,
        source_weight_version=source_version,
    )


def test_temporal_batch_plan_is_collective_safe_and_topology_neutral() -> None:
    config = _estimator(
        local_bptt_probability=1.0,
        overshoot_probability=0.0,
        source_mask_probability=0.0,
    )
    seed = native_temporal_batch_seed(
        parent_seed=19,
        comparison_id="full-objective",
        optimizer_step=7,
        sample_keys=("episode-a/4", "episode-b/12"),
    )
    first = sample_temporal_batch_plan(
        config,
        seed=seed,
        state_ages=(4, 12),
        available_future_steps=(64, 32),
        optimizer_lags=(0, 4),
    )
    replay = sample_temporal_batch_plan(
        config,
        seed=seed,
        state_ages=(4, 12),
        available_future_steps=(64, 32),
        optimizer_lags=(0, 4),
    )

    assert first == replay
    assert first.digest == replay.digest
    assert len(first.digest) == 64
    assert first.local_bptt_steps in (2, 3, 4)
    assert first.overshoot_horizon is None
    assert not first.source_masked_branch
    assert config.metadata["auxiliary_sampling"] == FROZEN_AUXILIARY_SAMPLING

    bounded = sample_temporal_batch_plan(
        config,
        seed=seed,
        state_ages=(4, 12),
        available_future_steps=(64, 0),
        optimizer_lags=(0, 0),
    )
    assert bounded.local_bptt_steps is None
    assert bounded.overshoot_horizon is None
    assert set(inspect.signature(sample_temporal_batch_plan).parameters) == {
        "config",
        "seed",
        "state_ages",
        "available_future_steps",
        "optimizer_lags",
    }


def test_temporal_auxiliary_mixture_is_exclusive_and_preserves_marginal_rates() -> None:
    config = _estimator(
        local_bptt_probability=0.25,
        overshoot_probability=0.20,
        source_mask_probability=0.15,
    )
    counts = {"local": 0, "overshoot": 0, "source": 0, "base": 0}
    sample_count = 4096
    for seed in range(sample_count):
        plan = sample_temporal_batch_plan(
            config,
            seed=seed,
            state_ages=(8, 9),
            available_future_steps=(64, 64),
            optimizer_lags=(0, 0),
        )
        active = (
            plan.local_bptt_steps is not None,
            plan.overshoot_horizon is not None,
            plan.source_masked_branch,
        )
        assert sum(active) <= 1
        selected = next(
            (
                name
                for name, enabled in zip(("local", "overshoot", "source"), active, strict=True)
                if enabled
            ),
            "base",
        )
        counts[selected] += 1

    expected = {"local": 0.25, "overshoot": 0.20, "source": 0.15, "base": 0.40}
    for name, probability in expected.items():
        assert counts[name] / sample_count == pytest.approx(probability, abs=0.025)


def test_temporal_auxiliary_mixture_rejects_overcommitted_probabilities() -> None:
    with pytest.raises(ValueError, match="sum to at most one"):
        _estimator(
            local_bptt_probability=0.5,
            overshoot_probability=0.3,
            source_mask_probability=0.3,
        )

    with pytest.raises(ValueError, match="at most one sparse auxiliary"):
        TemporalBatchPlan(
            seed=7,
            state_ages=(0,),
            local_bptt_steps=4,
            overshoot_horizon=None,
            source_masked_branch=True,
        )


def test_temporal_auxiliary_mixture_prevents_the_fresh20_step18_collision() -> None:
    plan = sample_temporal_batch_plan(
        _estimator(
            local_bptt_probability=0.10,
            overshoot_probability=0.05,
            source_mask_probability=0.10,
        ),
        seed=5766748332167034426,
        state_ages=(2, 2),
        available_future_steps=(64, 64),
        optimizer_lags=(0, 0),
    )

    assert plan.local_bptt_steps in (2, 3, 4)
    assert plan.overshoot_horizon is None
    assert not plan.source_masked_branch


def test_fresh60_step60_replays_the_exact_four_frame_local_window() -> None:
    plan = sample_temporal_batch_plan(
        _estimator(
            local_bptt_probability=0.10,
            overshoot_probability=0.05,
            source_mask_probability=0.10,
        ),
        seed=1849574444998761719,
        state_ages=(7, 7),
        available_future_steps=(64, 64),
        optimizer_lags=(0, 0),
    )

    assert plan.local_bptt_steps == 4
    assert plan.overshoot_horizon is None
    assert not plan.source_masked_branch


def test_temporal_target_schedule_is_invariant_to_legal_lane_metadata() -> None:
    """Cache coverage built at lag zero must contain every runtime target key."""

    config = _estimator(
        local_bptt_probability=0.3,
        overshoot_probability=0.3,
        source_mask_probability=0.3,
    )
    for seed in range(256):
        baseline = sample_temporal_batch_plan(
            config,
            seed=seed,
            state_ages=(0, 0),
            available_future_steps=(64, 32),
            optimizer_lags=(0, 0),
        )
        expected_schedule = (
            baseline.local_bptt_steps,
            baseline.overshoot_horizon,
            baseline.source_masked_branch,
        )
        for state_ages, optimizer_lags in (
            ((1, 99), (0, config.maximum_optimizer_lag)),
            ((10_000, 3), (config.maximum_optimizer_lag, 1)),
        ):
            runtime = sample_temporal_batch_plan(
                config,
                seed=seed,
                state_ages=state_ages,
                available_future_steps=(64, 32),
                optimizer_lags=optimizer_lags,
            )
            assert (
                runtime.local_bptt_steps,
                runtime.overshoot_horizon,
                runtime.source_masked_branch,
            ) == expected_schedule


def test_temporal_contract_has_no_state_changing_replay_controls() -> None:
    forbidden = {
        "refresh_probability",
        "refresh_after_optimizer_lag",
        "maximum_recompute_gap",
        "refresh_mask",
        "recompute_gaps",
    }
    assert not forbidden.intersection(inspect.signature(TemporalEstimatorConfig).parameters)
    assert not forbidden.intersection(TemporalEstimatorConfig.__dataclass_fields__)


def test_measured_cost_keeps_one_frame_path_and_sparse_auxiliaries_explicit() -> None:
    profile = TemporalCostProfile(
        full_step_seconds=20.0,
        row_step_seconds=0.01,
        source_masked_seconds=18.0,
    )
    workload = TemporalWorkload(
        local_extra_full_steps=0.4,
        prior_row_steps=0.1 * 127 / 7,
        source_masked_steps=0.1,
    )
    expected = 20.0 * 1.4 + 0.01 * 0.1 * 127 / 7 + 18.0 * 0.1
    assert profile.estimated_seconds(workload) == pytest.approx(expected)
    with pytest.raises(TypeError, match="measured TemporalWorkload"):
        profile.estimated_seconds(_estimator())  # type: ignore[arg-type]


def test_lane_advances_only_after_exact_optimizer_commit_and_clones_state() -> None:
    bank = _lane_bank()
    state = _state(1)
    transaction = bank.stage(
        0,
        state,
        _stamp(0, age=0, optimizer=10),
        reset=True,
    )
    assert len(bank) == 0
    with pytest.raises(NativeLaneError, match="immediately successful"):
        bank.commit_after_optimizer(transaction, successful_optimizer_step=10)
    assert len(bank) == 0
    bank.commit_after_optimizer(transaction, successful_optimizer_step=11)
    assert len(bank) == 1
    state.rows.add_(100)
    read = bank.read(
        0,
        episode_key="episode-a",
        next_frame_index=1,
        optimizer_step=11,
        source_weight_version=3,
    )
    assert read is not None
    assert torch.equal(read.state.rows, torch.ones(1, 2, 4))
    read.state.rows.add_(50)
    reread = bank.read(
        0,
        episode_key="episode-a",
        next_frame_index=1,
        optimizer_step=11,
        source_weight_version=3,
    )
    assert reread is not None
    assert torch.equal(reread.state.rows, torch.ones(1, 2, 4))


def test_lane_abort_continuity_staleness_and_source_version_fail_closed() -> None:
    bank = _lane_bank()
    first = bank.stage(0, _state(1), _stamp(0, age=0, optimizer=0), reset=True)
    bank.commit_after_optimizer(first, successful_optimizer_step=1)
    staged = bank.stage(0, _state(2), _stamp(1, age=1, optimizer=1), reset=False)
    bank.abort(staged)
    read = bank.read(
        0,
        episode_key="episode-a",
        next_frame_index=1,
        optimizer_step=1,
        source_weight_version=3,
    )
    assert read is not None and read.stamp.frame_index == 0
    with pytest.raises(NativeLaneError, match="contiguous"):
        bank.read(
            0,
            episode_key="episode-a",
            next_frame_index=2,
            optimizer_step=1,
            source_weight_version=3,
        )
    with pytest.raises(NativeLaneError, match="source-mixture"):
        bank.read(
            0,
            episode_key="episode-a",
            next_frame_index=1,
            optimizer_step=1,
            source_weight_version=4,
        )
    with pytest.raises(NativeLaneError, match="staleness"):
        bank.read(
            0,
            episode_key="episode-a",
            next_frame_index=1,
            optimizer_step=9,
            source_weight_version=3,
        )


def test_lane_batch_commit_and_abort_are_all_or_none() -> None:
    bank = _lane_bank()
    first = bank.stage(0, _state(1), _stamp(0, age=0, optimizer=4), reset=True)
    second = bank.stage(
        1,
        _state(2),
        _stamp(0, age=0, optimizer=4, episode="episode-b"),
        reset=True,
    )
    forged = type(second)(
        token=second.token + 100,
        lane_id=second.lane_id,
        state=second.state,
        stamp=second.stamp,
        reset=second.reset,
    )
    with pytest.raises(NativeLaneError, match="unknown"):
        bank.commit_batch_after_optimizer(
            (first, forged),
            successful_optimizer_step=5,
        )
    assert len(bank) == 0
    bank.abort_batch((first, second))
    assert len(bank) == 0

    first = bank.stage(0, _state(3), _stamp(0, age=0, optimizer=5), reset=True)
    second = bank.stage(
        1,
        _state(4),
        _stamp(0, age=0, optimizer=5, episode="episode-b"),
        reset=True,
    )
    bank.commit_batch_after_optimizer((first, second), successful_optimizer_step=6)
    assert len(bank) == 2


@pytest.mark.parametrize("operation", ["commit", "abort"])
def test_lane_batch_publication_is_atomic_under_index_fault(operation: str) -> None:
    bank = _lane_bank()
    first = bank.stage(0, _state(1), _stamp(0, age=0, optimizer=4), reset=True)
    second = bank.stage(
        1,
        _state(2),
        _stamp(0, age=0, optimizer=4, episode="episode-b"),
        reset=True,
    )
    bank._pending_lanes.remove(second.lane_id)
    with pytest.raises(NativeLaneError, match="internally inconsistent"):
        if operation == "commit":
            bank.commit_batch_after_optimizer((first, second), successful_optimizer_step=5)
        else:
            bank.abort_batch((first, second))
    assert len(bank) == 0
    assert bank._pending == {first.token: first, second.token: second}


def test_lane_predecessor_is_exact_checkpointed_lineage() -> None:
    bank = _lane_bank()
    first = bank.stage(0, _state(1), _stamp(0, age=0, optimizer=0), reset=True)
    bank.commit_after_optimizer(first, successful_optimizer_step=1)
    assert (
        bank.read_predecessor(
            0,
            episode_key="episode-a",
            next_frame_index=1,
            source_weight_version=3,
        )
        is None
    )

    second = bank.stage(0, _state(2), _stamp(1, age=1, optimizer=1), reset=False)
    bank.commit_after_optimizer(second, successful_optimizer_step=2)
    predecessor = bank.read_predecessor(
        0,
        episode_key="episode-a",
        next_frame_index=2,
        source_weight_version=3,
    )
    assert predecessor is not None
    torch.testing.assert_close(predecessor.rows, _state(1).rows)
    predecessor.rows.add_(99)

    restored = NativeTrainingLaneBank.deserialize(bank.config, bank.serialize())
    restored_predecessor = restored.read_predecessor(
        0,
        episode_key="episode-a",
        next_frame_index=2,
        source_weight_version=3,
    )
    assert restored_predecessor is not None
    torch.testing.assert_close(restored_predecessor.rows, _state(1).rows)


def test_lane_snapshot_is_deterministic_exact_and_forbidden_while_pending() -> None:
    bank = _lane_bank()
    first = bank.stage(
        7,
        _state(2),
        _stamp(0, age=0, optimizer=3),
        reset=True,
        row_bindings=(("object/a", 1),),
    )
    with pytest.raises(NativeLaneError, match="pending"):
        bank.serialize()
    bank.commit_after_optimizer(first, successful_optimizer_step=4)
    payload = bank.serialize()
    assert json.loads(payload)["version"] == 4
    assert payload == bank.serialize()
    restored = NativeTrainingLaneBank.deserialize(bank.config, payload)
    assert restored.serialize() == payload
    assert restored.digest == bank.digest
    read = restored.read(
        7,
        episode_key="episode-a",
        next_frame_index=1,
        optimizer_step=4,
        source_weight_version=3,
    )
    assert read is not None and read.row_bindings == (("object/a", 1),)


def test_lane_bindings_are_episode_local_monotonic_and_optimizer_transactional() -> None:
    bank = _lane_bank()
    first = bank.stage(
        0,
        _state(1),
        _stamp(0, age=0, optimizer=0),
        reset=True,
        row_bindings=(("object/a", 1),),
    )
    bank.commit_after_optimizer(first, successful_optimizer_step=1)

    with pytest.raises(NativeLaneError, match="removed or rebound"):
        bank.stage(
            0,
            _state(2),
            _stamp(1, age=1, optimizer=1),
            reset=False,
            row_bindings=(("object/a", 0),),
        )
    second = bank.stage(
        0,
        _state(2),
        _stamp(1, age=1, optimizer=1),
        reset=False,
        row_bindings=(("object/a", 1), ("object/b", 0)),
    )
    bank.abort(second)
    read = bank.read(
        0,
        episode_key="episode-a",
        next_frame_index=1,
        optimizer_step=1,
        source_weight_version=3,
    )
    assert read is not None and read.row_bindings == (("object/a", 1),)

    reset = bank.stage(
        0,
        _state(3),
        _stamp(0, age=0, optimizer=1, episode="episode-new"),
        reset=True,
        row_bindings=(("object/a", 0),),
    )
    bank.commit_after_optimizer(reset, successful_optimizer_step=2)
    read = bank.read(
        0,
        episode_key="episode-new",
        next_frame_index=1,
        optimizer_step=2,
        source_weight_version=3,
    )
    assert read is not None and read.row_bindings == (("object/a", 0),)


def test_lane_state_machine_randomized_commit_abort_and_round_trip() -> None:
    generator = random.Random(729_031)
    config = NativeLaneConfig(
        model_digest="lingbot-native-randomized-test",
        schema_digest="rows-v1",
        capacity=2,
        host_width=4,
        maximum_optimizer_lag=10_000,
    )
    bank = NativeTrainingLaneBank(config)
    expected: dict[int, tuple[str, int, int, torch.Tensor]] = {}
    optimizer_step = 0
    commits = 0
    aborts = 0

    for operation_index in range(256):
        lane_id = generator.randrange(4)
        previous = expected.get(lane_id)
        reset = previous is None or generator.random() < 0.2
        if reset:
            episode = f"episode-{lane_id}-{operation_index}"
            frame = age = 0
        else:
            episode, frame, age, _previous_rows = previous
            frame += 1
            age += 1
        rows = torch.full((1, 2, 4), float(operation_index + 1))
        transaction = bank.stage(
            lane_id,
            NativePosteriorState(rows),
            NativeLaneStamp(
                episode_key=episode,
                frame_index=frame,
                state_age=age,
                producer_optimizer_step=optimizer_step,
                source_weight_version=3,
            ),
            reset=reset,
        )
        with pytest.raises(NativeLaneError, match="pending"):
            bank.serialize()

        before = None if previous is None else bank._records[lane_id].state.rows.clone()
        snapshot_before = None
        if not bank._records and previous is None:
            snapshot_before = None
        else:
            # Abort must restore the exact pre-stage bytes, not only equivalent values.
            bank.abort(transaction)
            snapshot_before = bank.serialize()
            transaction = bank.stage(
                lane_id,
                NativePosteriorState(rows),
                NativeLaneStamp(
                    episode_key=episode,
                    frame_index=frame,
                    state_age=age,
                    producer_optimizer_step=optimizer_step,
                    source_weight_version=3,
                ),
                reset=reset,
            )

        if generator.random() < 0.25:
            bank.abort(transaction)
            aborts += 1
            if snapshot_before is not None:
                assert bank.serialize() == snapshot_before
            if previous is not None:
                torch.testing.assert_close(bank._records[lane_id].state.rows, before)
            continue

        prior_rows = None if previous is None or reset else previous[3]
        bank.commit_after_optimizer(
            transaction,
            successful_optimizer_step=optimizer_step + 1,
        )
        optimizer_step += 1
        commits += 1
        expected[lane_id] = (episode, frame, age, rows.clone())
        current = bank.read(
            lane_id,
            episode_key=episode,
            next_frame_index=frame + 1,
            optimizer_step=optimizer_step,
            source_weight_version=3,
        )
        assert current is not None
        torch.testing.assert_close(current.state.rows, rows)
        predecessor = bank.read_predecessor(
            lane_id,
            episode_key=episode,
            next_frame_index=frame + 1,
            source_weight_version=3,
        )
        if prior_rows is None:
            assert predecessor is None
        else:
            assert predecessor is not None
            torch.testing.assert_close(predecessor.rows, prior_rows)
        encoded = bank.serialize()
        bank = NativeTrainingLaneBank.deserialize(config, encoded)
        assert bank.serialize() == encoded

    assert commits > 150
    assert aborts > 30


def test_lane_snapshot_v2_migrates_current_and_predecessor_lineage_to_v4() -> None:
    bank = _lane_bank()
    first = bank.stage(0, _state(1), _stamp(0, age=0, optimizer=0), reset=True)
    bank.commit_after_optimizer(first, successful_optimizer_step=1)
    second = bank.stage(0, _state(2), _stamp(1, age=1, optimizer=1), reset=False)
    bank.commit_after_optimizer(second, successful_optimizer_step=2)

    legacy = json.loads(bank.serialize())
    legacy["version"] = 2
    for collection in (legacy["records"], legacy["history"]):
        for record in collection:
            record.pop("row_bindings")
            record["stamp"]["last_reconstruction_frame"] = 0

    restored = NativeTrainingLaneBank.deserialize(
        bank.config,
        json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode(),
    )
    current = restored.read(
        0,
        episode_key="episode-a",
        next_frame_index=2,
        optimizer_step=2,
        source_weight_version=3,
    )
    predecessor = restored.read_predecessor(
        0,
        episode_key="episode-a",
        next_frame_index=2,
        source_weight_version=3,
    )
    assert current is not None and predecessor is not None
    torch.testing.assert_close(current.state.rows, _state(2).rows)
    torch.testing.assert_close(predecessor.rows, _state(1).rows)
    assert json.loads(restored.serialize())["version"] == 4


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload: payload.pop("records"), "top-level schema"),
        (lambda payload: payload.update({"undeclared": True}), "top-level schema"),
        (lambda payload: payload.update({"records": {}}), "records must be a list"),
    ),
)
def test_lane_snapshot_rejects_non_exact_top_level_schema(mutation, message: str) -> None:
    bank = _lane_bank()
    transaction = bank.stage(0, _state(1), _stamp(0, age=0, optimizer=0), reset=True)
    bank.commit_after_optimizer(transaction, successful_optimizer_step=1)
    payload = json.loads(bank.serialize())
    mutation(payload)
    with pytest.raises(ValueError, match=message):
        NativeTrainingLaneBank.deserialize(bank.config, json.dumps(payload).encode())


def test_lane_snapshot_rejects_non_exact_record_and_stamp_schema() -> None:
    bank = _lane_bank()
    transaction = bank.stage(0, _state(1), _stamp(0, age=0, optimizer=0), reset=True)
    bank.commit_after_optimizer(transaction, successful_optimizer_step=1)
    payload = json.loads(bank.serialize())
    payload["records"][0]["undeclared"] = 1
    with pytest.raises(ValueError, match="record has an incompatible schema"):
        NativeTrainingLaneBank.deserialize(bank.config, json.dumps(payload).encode())
    payload = json.loads(bank.serialize())
    payload["records"][0]["stamp"]["state_age"] = False
    with pytest.raises(ValueError, match="counters must be non-negative integers"):
        NativeTrainingLaneBank.deserialize(bank.config, json.dumps(payload).encode())


def test_layerwise_lane_commit_predecessor_resume_and_legacy_rejection() -> None:
    config = NativeLaneConfig(
        model_digest="layerwise-model",
        schema_digest="layerwise-stream",
        capacity=2,
        host_width=4,
        maximum_optimizer_lag=8,
        num_layers=3,
    )
    bank = NativeTrainingLaneBank(config)
    first = bank.stage(0, _layerwise_state(1), _stamp(0, age=0, optimizer=0), reset=True)
    bank.commit_after_optimizer(first, successful_optimizer_step=1)
    second = bank.stage(0, _layerwise_state(2), _stamp(1, age=1, optimizer=1), reset=False)
    bank.commit_after_optimizer(second, successful_optimizer_step=2)
    read = bank.read(
        0,
        episode_key="episode-a",
        next_frame_index=2,
        optimizer_step=2,
        source_weight_version=3,
    )
    assert read is not None
    assert isinstance(read.state, NativeLayerwisePosteriorState)
    assert torch.equal(read.state.layer_rows, _layerwise_state(2).layer_rows)
    predecessor = bank.read_predecessor(
        0,
        episode_key="episode-a",
        next_frame_index=2,
        source_weight_version=3,
    )
    assert isinstance(predecessor, NativeLayerwisePosteriorState)
    assert torch.equal(predecessor.layer_rows, _layerwise_state(1).layer_rows)
    snapshot = bank.serialize()
    restored = NativeTrainingLaneBank.deserialize(config, snapshot)
    assert restored.serialize() == snapshot
    legacy_config = NativeLaneConfig(
        model_digest="layerwise-model",
        schema_digest="layerwise-stream",
        capacity=2,
        host_width=4,
        maximum_optimizer_lag=8,
    )
    with pytest.raises(ValueError, match="lane contract"):
        NativeTrainingLaneBank.deserialize(legacy_config, snapshot)


def test_addressed_layerwise_lane_preserves_routing_receipt_across_resume() -> None:
    config = NativeLaneConfig(
        model_digest="addressed-layerwise-model",
        schema_digest="addressed-layerwise-stream",
        capacity=2,
        host_width=4,
        maximum_optimizer_lag=8,
        num_layers=3,
        addressed_architecture_identity=LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
        episode_address_codebook_sha256="a" * 64,
    )
    bank = NativeTrainingLaneBank(config)
    first_state = _addressed_layerwise_state(1)
    first = bank.stage(0, first_state, _stamp(0, age=0, optimizer=0), reset=True)
    bank.commit_after_optimizer(first, successful_optimizer_step=1)
    second_state = _addressed_layerwise_state(2)
    second = bank.stage(0, second_state, _stamp(1, age=1, optimizer=1), reset=False)
    bank.commit_after_optimizer(second, successful_optimizer_step=2)

    snapshot = bank.serialize()
    assert json.loads(snapshot)["version"] == 6
    restored = NativeTrainingLaneBank.deserialize(config, snapshot)
    assert restored.serialize() == snapshot
    read = restored.read(
        0,
        episode_key="episode-a",
        next_frame_index=2,
        optimizer_step=2,
        source_weight_version=3,
    )
    assert read is not None
    assert isinstance(read.state, AddressedLayerwisePosteriorState)
    assert read.state.address_receipt == second_state.address_receipt
    assert read.state.architecture_identity == LINGBOT_TASK_QUERY_OBJECT_VALUE_READ
    predecessor = restored.read_predecessor(
        0,
        episode_key="episode-a",
        next_frame_index=2,
        source_weight_version=3,
    )
    assert isinstance(predecessor, AddressedLayerwisePosteriorState)
    assert predecessor.address_receipt == first_state.address_receipt


def test_paired_videomt_lane_is_atomic_across_commit_resume_and_snapshot() -> None:
    config = NativeLaneConfig(
        model_digest="paired-videomt-model",
        schema_digest="paired-videomt-stream",
        capacity=2,
        host_width=4,
        maximum_optimizer_lag=8,
        num_layers=3,
        paired_source_width=6,
        paired_architecture_identity="native-videomt-query-posterior/v1",
        paired_source_dtype=torch.float32,
        dtype=torch.bfloat16,
    )
    bank = NativeTrainingLaneBank(config)
    first_state = _paired_videomt_state(1)
    first = bank.stage(0, first_state, _stamp(0, age=0, optimizer=0), reset=True)
    assert isinstance(first.state, NativeVidEoMTPairedPosteriorState)
    assert not first.state.layer_rows.requires_grad
    assert not first.state.source_queries.requires_grad
    assert first.state.layer_rows.data_ptr() != first_state.layer_rows.data_ptr()
    assert first.state.source_queries.data_ptr() != first_state.source_queries.data_ptr()
    bank.commit_after_optimizer(first, successful_optimizer_step=1)

    second_state = _paired_videomt_state(2)
    second = bank.stage(0, second_state, _stamp(1, age=1, optimizer=1), reset=False)
    bank.commit_after_optimizer(second, successful_optimizer_step=2)
    snapshot = bank.serialize()
    assert json.loads(snapshot)["version"] == 7

    restored = NativeTrainingLaneBank.deserialize(config, snapshot)
    assert restored.serialize() == snapshot
    read = restored.read(
        0,
        episode_key="episode-a",
        next_frame_index=2,
        optimizer_step=2,
        source_weight_version=3,
    )
    assert read is not None
    assert isinstance(read.state, NativeVidEoMTPairedPosteriorState)
    torch.testing.assert_close(read.state.layer_rows, second_state.layer_rows)
    torch.testing.assert_close(read.state.source_queries, second_state.source_queries)
    predecessor = restored.read_predecessor(
        0,
        episode_key="episode-a",
        next_frame_index=2,
        source_weight_version=3,
    )
    assert isinstance(predecessor, NativeVidEoMTPairedPosteriorState)
    torch.testing.assert_close(predecessor.layer_rows, first_state.layer_rows)
    torch.testing.assert_close(predecessor.source_queries, first_state.source_queries)


def test_paired_videomt_lane_rejects_partial_or_unpaired_state_contracts() -> None:
    common = {
        "model_digest": "paired-videomt-model",
        "schema_digest": "paired-videomt-stream",
        "capacity": 2,
        "host_width": 4,
        "maximum_optimizer_lag": 8,
        "num_layers": 3,
        "dtype": torch.bfloat16,
    }
    with pytest.raises(ValueError, match="declared together"):
        NativeLaneConfig(**common, paired_source_width=6)
    config = NativeLaneConfig(
        **common,
        paired_source_width=6,
        paired_architecture_identity="native-videomt-query-posterior/v1",
    )
    with pytest.raises(ValueError, match="atomic paired posterior"):
        NativeTrainingLaneBank(config).stage(
            0,
            NativeLayerwisePosteriorState(
                torch.zeros(1, 3, 2, 4, dtype=torch.bfloat16)
            ),
            _stamp(0, age=0, optimizer=0),
            reset=True,
        )


def test_addressed_lane_contract_rejects_missing_or_mismatched_routing_identity() -> None:
    common = {
        "model_digest": "addressed-layerwise-model",
        "schema_digest": "addressed-layerwise-stream",
        "capacity": 2,
        "host_width": 4,
        "maximum_optimizer_lag": 8,
        "num_layers": 3,
    }
    with pytest.raises(ValueError, match="declared together"):
        NativeLaneConfig(
            **common,
            addressed_architecture_identity=LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
        )
    config = NativeLaneConfig(
        **common,
        addressed_architecture_identity=LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
        episode_address_codebook_sha256="a" * 64,
    )
    bank = NativeTrainingLaneBank(config)
    with pytest.raises(ValueError, match="routing contract"):
        bank.stage(
            0,
            _addressed_layerwise_state(1, codebook_sha256="b" * 64),
            _stamp(0, age=0, optimizer=0),
            reset=True,
        )
    with pytest.raises(ValueError, match="routing contract"):
        bank.stage(
            0,
            _addressed_layerwise_state(1, architecture_identity="wrong-identity"),
            _stamp(0, age=0, optimizer=0),
            reset=True,
        )
    with pytest.raises(ValueError, match="requires addressed posterior"):
        bank.stage(
            0,
            _layerwise_state(1),
            _stamp(0, age=0, optimizer=0),
            reset=True,
        )


def test_historical_layerwise_lane_rejects_addressed_snapshot_and_state() -> None:
    historical = NativeLaneConfig(
        model_digest="layerwise-model",
        schema_digest="layerwise-stream",
        capacity=2,
        host_width=4,
        maximum_optimizer_lag=8,
        num_layers=3,
    )
    with pytest.raises(ValueError, match="historical lane"):
        NativeTrainingLaneBank(historical).stage(
            0,
            _addressed_layerwise_state(1),
            _stamp(0, age=0, optimizer=0),
            reset=True,
        )

    addressed = NativeLaneConfig(
        model_digest="layerwise-model",
        schema_digest="layerwise-stream",
        capacity=2,
        host_width=4,
        maximum_optimizer_lag=8,
        num_layers=3,
        addressed_architecture_identity=LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
        episode_address_codebook_sha256="a" * 64,
    )
    bank = NativeTrainingLaneBank(addressed)
    transaction = bank.stage(
        0,
        _addressed_layerwise_state(1),
        _stamp(0, age=0, optimizer=0),
        reset=True,
    )
    bank.commit_after_optimizer(transaction, successful_optimizer_step=1)
    with pytest.raises(ValueError, match="lane contract"):
        NativeTrainingLaneBank.deserialize(historical, bank.serialize())


def test_layerwise_lane_transaction_detaches_the_optimizer_segment_boundary() -> None:
    config = NativeLaneConfig(
        model_digest="layerwise-model",
        schema_digest="layerwise-stream",
        capacity=2,
        host_width=4,
        maximum_optimizer_lag=8,
        num_layers=3,
    )
    bank = NativeTrainingLaneBank(config)
    source = torch.randn(1, 3, 2, 4, requires_grad=True)
    transaction = bank.stage(
        0,
        NativeLayerwisePosteriorState(source),
        _stamp(0, age=0, optimizer=0),
        reset=True,
    )
    assert isinstance(transaction.state, NativeLayerwisePosteriorState)
    assert not transaction.state.layer_rows.requires_grad
    assert transaction.state.layer_rows.data_ptr() != source.data_ptr()
    bank.commit_after_optimizer(transaction, successful_optimizer_step=1)
    read = bank.read(
        0,
        episode_key="episode-a",
        next_frame_index=1,
        optimizer_step=1,
        source_weight_version=3,
    )
    assert read is not None
    assert isinstance(read.state, NativeLayerwisePosteriorState)
    assert not read.state.layer_rows.requires_grad


def test_row_only_predictive_rollout_uses_shared_host_graph() -> None:
    graph = LingBotNativeGraph(
        LingBotNativeGraphConfig(
            capacity=2,
            host_width=4,
            executed_action_dim=2,
            num_layers=2,
            maximum_control_tokens=1,
            predictive_target_widths=(("dino_video", 4),),
        )
    ).train()
    policy = _ToyOfficialPolicy(graph)
    stepper = LingBotNativePriorStepper(policy, graph)
    controls = tuple(_control(float(index)) for index in range(4))
    initial_rows = torch.randn(1, 2, 4, requires_grad=True)
    initial = NativePosteriorState(initial_rows)
    request = NativePredictionRequest(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.FUTURE,
        route_ids=torch.zeros(1, 1, dtype=torch.long),
        horizons=torch.full((1, 1), 4, dtype=torch.long),
        addresses=torch.empty(1, 1, 0),
        valid=torch.ones(1, 1, dtype=torch.bool),
    )
    rollout = rollout_native_prior_prediction(
        stepper,
        initial,
        controls,
        request=request,
        target_name="dino_video",
    )

    assert rollout.target_name == "dino_video"
    rollout.prediction.square().mean().backward()
    assert graph.prediction_horizon_projection.weight.grad is not None
    assert graph.prediction_horizon_projection.weight.grad.abs().sum() > 0
    assert initial_rows.grad is not None
    assert initial_rows.grad.abs().sum() > 0
    assert graph.object_queries.grad is None or not graph.object_queries.grad.count_nonzero()
    assert graph.role_embeddings.grad is not None
    assert graph.role_embeddings.grad.abs().sum() > 0
    assert graph.predictive_readout("dino_video").weight.grad is not None
    assert not isinstance(stepper, torch.nn.Module)
    assert vars(stepper) == {"policy": policy, "graph": graph}
    assert (
        "activation_checkpointing"
        not in inspect.signature(rollout_native_prior_prediction).parameters
    )
