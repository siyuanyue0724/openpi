from __future__ import annotations

from types import SimpleNamespace

from scripts.picf_stage_calvin_partial_cache import _required_step_ids


def test_required_step_ids_includes_per_rank_warmup_windows() -> None:
    source = SimpleNamespace(
        unroll_steps=2,
        window_index=[
            (0, 100),  # warmup for rank 0
            (0, 200),  # warmup for rank 1
            (1, 300),
            (1, 400),
        ],
    )
    source.__len__ = lambda: len(source.window_index)  # type: ignore[attr-defined]

    class _Wrapper:
        def __len__(self) -> int:
            return len(source.window_index)

        @property
        def window_index(self):
            return source.window_index

        @property
        def unroll_steps(self):
            return source.unroll_steps

    step_ids, summary = _required_step_ids(
        source=_Wrapper(),
        seed=42,
        ranks=[0, 1],
        steps_per_rank=0,
    )

    # Even with zero sampled train windows, the script must stage rank warmup windows
    # because picf_core_train materializes lazy modules with source.window(rank).
    assert {100, 101, 102}.issubset(step_ids)
    assert {200, 201, 202}.issubset(step_ids)
    assert summary[0]["warmup_flat_index"] == 0
    assert summary[1]["warmup_flat_index"] == 1
