from scripts.verify_picf_owm_contract import main
from scripts.verify_picf_owm_contract import run_checks


def test_verify_picf_owm_contract_all_checks_pass() -> None:
    checks = run_checks()

    assert checks
    assert all(check.ok for check in checks), [check for check in checks if not check.ok]


def test_verify_picf_owm_contract_cli_json(capsys) -> None:
    assert main(["--json"]) == 0
    output = capsys.readouterr().out
    assert '"ok": true' in output
    assert "training_uses_next_posterior_teacher" in output
