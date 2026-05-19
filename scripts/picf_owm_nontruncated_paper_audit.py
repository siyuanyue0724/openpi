#!/usr/bin/env python3
"""Strict paper-code and no-shortcut audit for PICF object binding.

This audit intentionally checks cross-repository provenance, PICF dataflow, and
training/loss boundaries. It is designed to catch two opposite mistakes:

1. under-implementation: only auditing raw cosine while claiming IsSameObject;
2. overreach: turning weak geometry-derived labels into an online identity loss.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = Path("/tmp/picf_paper_code_20260515")


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    detail: str
    evidence: list[str]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _refs(path: Path, *needles: str, limit: int = 12) -> list[str]:
    if not path.exists():
        return [f"{path}: missing"]
    lines = _read(path).splitlines()
    out: list[str] = []
    for needle in needles:
        for idx, line in enumerate(lines, start=1):
            if needle in line:
                try:
                    rel = path.relative_to(REPO_ROOT)
                except ValueError:
                    rel = path
                out.append(f"{rel}:{idx}: {line.strip()}")
                break
        if len(out) >= limit:
            break
    return out


def _git(path: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _provenance_matches(path: Path, *, remote: str, head: str) -> bool:
    if _git(path, "remote", "get-url", "origin") == remote and _git(path, "rev-parse", "HEAD") == head:
        return True
    manifest = path / ".picf_provenance.json"
    if not manifest.exists():
        return False
    try:
        data = json.loads(_read(manifest))
    except Exception:
        return False
    return str(data.get("remote")) == remote and str(data.get("head")) == head


def _has_all(text: str, needles: Iterable[str]) -> bool:
    return all(needle in text for needle in needles)


def run_checks() -> list[Check]:
    vit = PAPER_ROOT / "vit-object-binding"
    slotcontrast = PAPER_ROOT / "slotcontrast"
    vit_models = vit / "src" / "utils" / "models.py"
    vit_trainer = vit / "src" / "trainer.py"
    slot_losses = slotcontrast / "slotcontrast" / "losses.py"
    slot_license = slotcontrast / "LICENSE"

    probe = REPO_ROOT / "scripts" / "picf_owm_same_object_probe.py"
    trainer = REPO_ROOT / "scripts" / "picf_core_train.py"
    config = REPO_ROOT / "src" / "openpi" / "picf" / "core" / "config.py"
    pipeline = REPO_ROOT / "src" / "openpi" / "picf" / "core" / "pipeline.py"
    readme = REPO_ROOT / "src" / "openpi" / "picf" / "README_v2.2.md"
    vnext = REPO_ROOT / "docs" / "PICF_AQR_OWM_VNEXT_FULL_MODEL_DESIGN_20260515_TEMP.md"
    tests = REPO_ROOT / "scripts" / "picf_owm_same_object_probe_test.py"

    vit_text = _read(vit_models) if vit_models.exists() else ""
    vit_trainer_text = _read(vit_trainer) if vit_trainer.exists() else ""
    slot_text = _read(slot_losses) if slot_losses.exists() else ""
    probe_text = _read(probe)
    trainer_text = _read(trainer)
    config_text = _read(config)
    pipeline_text = _read(pipeline)
    readme_text = _read(readme)
    vnext_text = _read(vnext)
    tests_text = _read(tests)

    checks: list[Check] = []
    checks.append(
        Check(
            "external_vit_binding_code_is_inspected_and_not_copied",
            vit.exists()
            and _provenance_matches(
                vit,
                remote="https://github.com/liyihao0302/vit-object-binding.git",
                head="014c66b45ea262f9b6eec83ff388a1e1c10dfcaa",
            )
            and not any(vit.glob("LICENSE*"))
            and _has_all(
                vit_text,
                ["DiagonalQuadraticProbe", "QuadraticProbe", "QuadraticFixedRankProbe", "forward_pairwise"],
            )
            and _has_all(vit_trainer_text, ["BCEWithLogitsLoss", "labels_pairwise", "compute_batch_pairwise_similarity"]),
            "The object-binding code snapshot must be explicitly inspected. Since no LICENSE exists, PICF must reimplement equations/protocols rather than copy source.",
            _refs(vit_models, "DiagonalQuadraticProbe", "QuadraticProbe", "QuadraticFixedRankProbe")
            + _refs(vit_trainer, "BCEWithLogitsLoss", "labels_pairwise"),
        )
    )
    checks.append(
        Check(
            "slotcontrast_code_is_licensed_and_used_only_as_design_evidence",
            slotcontrast.exists()
            and _provenance_matches(
                slotcontrast,
                remote="https://github.com/martius-lab/slotcontrast.git",
                head="55ec66dc02eeade630805789ef4a6c5df06f21ff",
            )
            and slot_license.exists()
            and "MIT License" in _read(slot_license)
            and _has_all(slot_text, ["Slot_Slot_Contrastive_Loss", "normalize", "batch_contrast", "CrossEntropyLoss"]),
            "SlotContrast is licensed and relevant, but PICF must keep native posterior-object-file losses rather than paste a standalone slot model.",
            _refs(slot_losses, "Slot_Slot_Contrastive_Loss", "batch_contrast", "CrossEntropyLoss") + _refs(slot_license, "MIT License"),
        )
    )
    checks.append(
        Check(
            "picf_probe_implements_full_quadratic_family",
            _has_all(
                probe_text,
                [
                    "class _DiagonalQuadraticProbe",
                    "class _LowRankQuadraticProbe",
                    "class _FullQuadraticProbe",
                    "0.5 * (xy + yx)",
                    "0.5 * (self.weight + self.weight.T)",
                    "--quadratic-probe",
                    "diag_quadratic",
                    "low_rank_quadratic",
                    "full_quadratic",
                    "all",
                ],
            ),
            "The PICF probe must not stop at raw cosine; it must include diagonal, low-rank, and full quadratic probes.",
            _refs(probe, "class _DiagonalQuadraticProbe", "class _LowRankQuadraticProbe", "class _FullQuadraticProbe", "--quadratic-probe"),
        )
    )
    checks.append(
        Check(
            "training_overlays_export_signatures_without_extra_forward",
            _has_all(
                trainer_text,
                [
                    "--anchor-overlay-dump-signatures",
                    "capture_anchor_overlay_signatures",
                    "dump_signatures=capture_anchor_overlay_signatures",
                    "support_signature",
                    "binding_signature",
                    "_anchor_overlay_snapshot_from_output",
                ],
            )
            and "_anchor_overlay_snapshot_from_output(" in trainer_text
            and "forward_train_transition" in trainer_text,
            "Training overlays must export the actual forward state's signatures and avoid a second side-effecting diagnostic forward.",
            _refs(trainer, "--anchor-overlay-dump-signatures", "capture_anchor_overlay_signatures", "dump_signatures=capture_anchor_overlay_signatures"),
        )
    )
    checks.append(
        Check(
            "binding_signature_is_runtime_binding_evidence_not_json_only",
            _has_all(
                pipeline_text,
                [
                    "self.binding_signature_proj",
                    "self.binding_quadratic_diag",
                    "self.binding_low_rank_left",
                    "def _binding_signature_quadratic_scores",
                    "def _support_binding_signature",
                    "binding_signature=obs_binding_signature",
                    "prev.binding_signature",
                    "obs.binding_signature",
                    "bind_embedding_signature_weight",
                    "bind_quadratic_signature_weight",
                    "bind_low_rank_signature_weight",
                    "innovation_decay",
                ],
            ),
            "Projected binding signatures and native quadratic same-object scores must feed posterior binding logits, not only diagnostics.",
            _refs(
                pipeline,
                "self.binding_signature_proj",
                "def _binding_signature_quadratic_scores",
                "def _support_binding_signature",
                "binding_signature=obs_binding_signature",
                "prev.binding_signature",
                "bind_embedding_signature_weight",
            ),
        )
    )
    checks.append(
        Check(
            "runtime_quadratic_binding_is_native_not_paper_code_copy",
            _has_all(
                config_text + "\n" + pipeline_text + "\n" + trainer_text,
                [
                    "bind_quadratic_signature_weight",
                    "bind_low_rank_signature_weight",
                    "binding_low_rank_signature_rank",
                    "self.binding_quadratic_diag",
                    "0.5 * (",
                    "self.binding_low_rank_left(prev_norm)",
                    "--bind-quadratic-signature-weight",
                    "--bind-low-rank-signature-weight",
                ],
            )
            and "from src.utils.models import" not in pipeline_text
            and "vit-object-binding" not in pipeline_text,
            "Runtime binding should natively reimplement diagonal/low-rank quadratic scoring and avoid importing unlicensed paper code.",
            _refs(
                config,
                "bind_quadratic_signature_weight",
                "binding_low_rank_signature_rank",
            )
            + _refs(pipeline, "self.binding_quadratic_diag", "self.binding_low_rank_left(prev_norm)")
            + _refs(trainer, "--bind-quadratic-signature-weight"),
        )
    )
    checks.append(
        Check(
            "weak_same_object_probe_is_not_online_training_loss",
            "import scripts.picf_owm_same_object_probe" not in trainer_text
            and "picf_owm_same_object_probe" not in pipeline_text
            and _has_all(readme_text, ["They are not task labels and not used as training loss", "diagnostic data export, not a model/loss change"]),
            "Weak adjacent-frame labels are allowed for offline audit only; online loss would be self-confirming without masks/tracklets.",
            _refs(readme, "not used as training loss", "diagnostic data export, not a model/loss change"),
        )
    )
    checks.append(
        Check(
            "docs_record_no_shortcut_boundary_and_remaining_limits",
            _has_all(
                readme_text + "\n" + vnext_text,
                [
                    "no LICENSE file was found",
                    "reimplements the published",
                    "The quadratic IsSameObject probe is an audit, not a training loss",
                    "Tracklet/proposal branches remain no-op on CALVIN unless the dataflow supplies",
                    "Ordinal",
                    "30k",
                ],
            ),
            "Reviewer-facing docs must show paper provenance, math, and non-negotiable limits.",
            _refs(readme, "Paper-code provenance", "not used as training loss")
            + _refs(vnext, "External code inspected", "Remaining non-negotiable boundaries"),
        )
    )
    checks.append(
        Check(
            "tests_cover_overlay_and_quadratic_probe",
            _has_all(
                tests_text,
                [
                    "test_same_object_probe_reads_training_anchor_overlays",
                    "test_same_object_probe_trains_quadratic_binding_probe",
                    "diag_quadratic",
                    "low_rank_quadratic",
                    "full_quadratic",
                ],
            ),
            "The audit must have tests for training overlays and the full quadratic probe family.",
            _refs(tests, "test_same_object_probe_reads_training_anchor_overlays", "test_same_object_probe_trains_quadratic_binding_probe"),
        )
    )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fail-on-fail", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    checks = run_checks()
    payload = {
        "pass": all(check.passed for check in checks),
        "checks": [
            {
                "name": check.name,
                "pass": check.passed,
                "detail": check.detail,
                "evidence": check.evidence,
            }
            for check in checks
        ],
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for check in checks:
            status = "PASS" if check.passed else "FAIL"
            print(f"{status} {check.name}: {check.detail}")
            for ev in check.evidence:
                print(f"  - {ev}")
        print(f"SUMMARY pass={sum(c.passed for c in checks)} fail={sum(not c.passed for c in checks)} total={len(checks)}")
    if args.fail_on_fail and not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
