from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "references/source_snapshots" / "videomt-025b9384214bf28cd90d51846464615dd4f443ac"
VENDORED = ROOT / "src/picf_next/_vendor/videomt"


def test_normative_videomt_sources_are_byte_identical() -> None:
    source_pairs = (
        ("videomt/modeling/backbone/videomt.py", "modeling/backbone/videomt.py"),
        ("videomt/modeling/backbone/vit.py", "modeling/backbone/vit.py"),
        ("videomt/modeling/backbone/scale_block.py", "modeling/backbone/scale_block.py"),
        ("videomt/modeling/matcher.py", "matcher.py"),
        ("videomt/data_video/augmentation.py", "data_video/augmentation.py"),
        (
            "videomt/modeling/two_stage_warmup_poly_schedule.py",
            "modeling/two_stage_warmup_poly_schedule.py",
        ),
        ("videomt/utils/misc.py", "utils/misc.py"),
        ("LICENSE", "LICENSE"),
    )
    for upstream_path, vendored_path in source_pairs:
        assert (UPSTREAM / upstream_path).read_bytes() == (VENDORED / vendored_path).read_bytes()


def test_criterion_diff_is_only_the_documented_import_adapter() -> None:
    upstream = (UPSTREAM / "videomt/criterion_videomt.py").read_text(encoding="utf-8")
    vendored = (VENDORED / "criterion_videomt.py").read_text(encoding="utf-8")
    normalized = vendored.replace(
        "from picf_next._vendor.videomt.utils.misc import is_dist_avail_and_initialized",
        "from videomt.utils.misc import is_dist_avail_and_initialized",
    )
    assert normalized.rstrip("\n") == upstream.rstrip("\n")
