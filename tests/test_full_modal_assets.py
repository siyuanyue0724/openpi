from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from picf_next.contracts import ContractError
from picf_next.full_modal_assets import FullModalAssetManifest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "references" / "full_modal_assets.json"


def _payload() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="ascii"))


def _write(tmp_path: Path, value: object) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="ascii")
    return path


def test_repository_full_modal_manifest_is_exact_and_fail_closed() -> None:
    manifest = FullModalAssetManifest.load(MANIFEST)

    assert tuple(asset.modality for asset in manifest.assets) == (
        "anytouch",
        "sonata",
        "vjepa",
    )
    assert manifest.asset("vjepa").model == "V-JEPA2.1 ViT-B 384"
    assert manifest.asset("sonata").architecture_upstream_commit is not None


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("allow_directory_scan", True),
        ("allow_unlisted_checkpoint", True),
        ("require_exact_sha256", False),
        ("require_strict_state_dict_load", False),
    ),
)
def test_manifest_rejects_permissive_selection_policy(
    tmp_path: Path, field: str, value: bool
) -> None:
    payload = _payload()
    payload["selection_policy"][field] = value

    with pytest.raises(ContractError, match="fail-closed"):
        FullModalAssetManifest.load(_write(tmp_path, payload))


def test_manifest_rejects_substituted_model_or_role(tmp_path: Path) -> None:
    payload = _payload()
    payload["assets"][0], payload["assets"][2] = (
        payload["assets"][2],
        payload["assets"][0],
    )
    with pytest.raises(ContractError, match="canonical modality order"):
        FullModalAssetManifest.load(_write(tmp_path, payload))

    payload = _payload()
    payload["assets"][2]["production_role"] = "generic_video_encoder"
    with pytest.raises(ContractError, match="production role changed"):
        FullModalAssetManifest.load(_write(tmp_path, payload))


def test_manifest_file_verification_rejects_wrong_bytes(tmp_path: Path) -> None:
    payload = deepcopy(_payload())
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"not-the-pinned-checkpoint")
    payload["assets"][0]["persistent_path"] = str(checkpoint)
    payload["assets"][0]["size_bytes"] = checkpoint.stat().st_size

    with pytest.raises(ContractError, match="SHA-256 changed"):
        FullModalAssetManifest.load(_write(tmp_path, payload), verify_files=True)


def test_manifest_rejects_abbreviated_upstream_commit(tmp_path: Path) -> None:
    payload = _payload()
    payload["assets"][0]["upstream_commit"] = "82c5677d"

    with pytest.raises(ContractError, match="full lowercase Git commit"):
        FullModalAssetManifest.load(_write(tmp_path, payload))
