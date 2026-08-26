"""Lightweight loss-only target locators for native CALVIN training."""

from __future__ import annotations

from dataclasses import dataclass

from picf_next.data.calvin import CalvinPhysicalSample, CalvinStatefulTransitionSample

CALVIN_PHYSICAL_SOURCE_FIELDS = frozenset(
    {"rgb_static", "depth_static", "rgb_gripper", "depth_gripper"}
)


@dataclass(frozen=True, slots=True)
class NativeCALVINStructuralTargetRequest:
    """Opaque locator that is never exposed to a deploy-visible model input."""

    sample_key: str
    episode_key: str
    task_key: str
    segment_index: int
    source_global_index: int
    source_sensor_sha256: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        strings = (self.sample_key, self.episode_key, self.task_key)
        if any(not isinstance(value, str) or not value for value in strings):
            raise ValueError("native CALVIN target identities cannot be empty")
        integers = (self.segment_index, self.source_global_index)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in integers
        ):
            raise ValueError("native CALVIN target indices must be non-negative integers")
        names = tuple(name for name, _digest in self.source_sensor_sha256)
        if set(names) != CALVIN_PHYSICAL_SOURCE_FIELDS or len(names) != len(set(names)):
            raise ValueError("native CALVIN target request has incomplete sensor hashes")
        if names != tuple(sorted(names)):
            raise ValueError("native CALVIN target sensor hashes must be sorted")
        for _name, digest in self.source_sensor_sha256:
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("native CALVIN target request sensor hash is invalid")

    @property
    def source_sensor_hash_by_field(self) -> dict[str, str]:
        return dict(self.source_sensor_sha256)


def native_calvin_structural_target_request(
    sample: CalvinStatefulTransitionSample | CalvinPhysicalSample,
) -> NativeCALVINStructuralTargetRequest:
    """Freeze a locator while deferring the heavy loss-side schema import."""

    if not isinstance(sample, CalvinStatefulTransitionSample | CalvinPhysicalSample):
        raise TypeError("native CALVIN targets require a typed transition sample")
    from picf_next.data.calvin_loss_targets import calvin_physical_source_hashes

    return NativeCALVINStructuralTargetRequest(
        sample_key=sample.sample_key,
        episode_key=sample.episode_key,
        task_key=sample.host_sample.task_key,
        segment_index=sample.record.task_index,
        source_global_index=sample.record.global_index,
        source_sensor_sha256=calvin_physical_source_hashes(sample.record),
    )
