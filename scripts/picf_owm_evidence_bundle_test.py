import json
from pathlib import Path

from scripts.picf_owm_evidence_bundle import build_bundle
from scripts.picf_owm_evidence_bundle import main


def test_build_owm_evidence_bundle_collects_args_metrics_and_diagnostics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    diag_dir = run_dir / "diagnostics" / "000010"
    diag_dir.mkdir(parents=True)
    (run_dir / "args.json").write_text(
        json.dumps(
            {
                "aqr_vjepa_temporal_mode": "last_two_tokens",
                "lambda_slot_jepa": 0.1,
                "unrelated": "ignore",
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"step": 1, "loss_slot_jepa": 0.5}),
                json.dumps({"step": 2, "loss_slot_jepa": 0.25, "evidence_cache_trust_mean": 0.8}),
            ]
        ),
        encoding="utf-8",
    )
    (diag_dir / "metadata.json").write_text(json.dumps({"step": 10}), encoding="utf-8")
    (diag_dir / "compare_grid.png").write_bytes(b"png")

    bundle = build_bundle(run_dir, tail=2)

    assert bundle["args_owm"] == {"aqr_vjepa_temporal_mode": "last_two_tokens", "lambda_slot_jepa": 0.1}
    assert bundle["latest_owm_metrics"]["loss_slot_jepa"] == 0.25
    assert bundle["latest_owm_metrics"]["evidence_cache_trust_mean"] == 0.8
    assert bundle["diagnostics"][0]["files"] == ["diagnostics/000010/compare_grid.png"]
    assert bundle["contract_verifier"]["ok"] is True
    assert any(check["name"] == "pipeline_cache_order_is_causal" for check in bundle["contract_verifier"]["checks"])
    assert bundle["audit_rules"]["posterior_authoritative"] is True


def test_owm_evidence_bundle_cli_writes_json(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 1, "loss_support_pred": 0.2}), encoding="utf-8")
    output = tmp_path / "bundle.json"

    assert main(["--run-dir", str(run_dir), "--output", str(output), "--tail", "1"]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["latest_owm_metrics"] == {"loss_support_pred": 0.2}
    printed = json.loads(capsys.readouterr().out)
    assert printed["output"] == str(output)
