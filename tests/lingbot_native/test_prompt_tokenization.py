from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from picf_next.contracts import ContractError
from picf_next.lingbot_native.prompt_tokenization import (
    CompletePromptTokenizationAudit,
    audit_complete_prompt_tokenization,
    format_lingbot_prompt,
    validate_distinct_prompt_tokenizations,
)


class _CharacterTokenizer:
    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is False
        assert messages[0]["role"] == "user"
        return f"<user>{messages[0]['content']}</user>"

    def __call__(
        self,
        prompts: list[str],
        *,
        padding: str | bool,
        truncation: bool,
        return_tensors: str,
        padding_side: str | None = None,
        max_length: int | None = None,
    ) -> dict[str, torch.Tensor]:
        assert len(prompts) == 1
        assert return_tensors == "pt"
        token_ids = [1001, *(ord(character) + 10 for character in prompts[0]), 1002]
        if truncation:
            assert padding == "max_length"
            assert padding_side == "right"
            assert max_length is not None
            token_ids = token_ids[:max_length]
        else:
            assert padding is False
            assert padding_side is None
            assert max_length is None
        attention_mask = [1] * len(token_ids)
        if max_length is not None:
            padding_count = max_length - len(token_ids)
            token_ids.extend([0] * padding_count)
            attention_mask.extend([0] * padding_count)
        return {
            "attention_mask": torch.tensor([attention_mask], dtype=torch.int64),
            "input_ids": torch.tensor([token_ids], dtype=torch.int64),
        }


class _MutatingBoundedTokenizer(_CharacterTokenizer):
    def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        result = super().__call__(prompts, **kwargs)
        if kwargs["truncation"] is True:
            result["input_ids"][0, 1] += 1
        return result


class _CollidingTokenizer(_CharacterTokenizer):
    def __call__(self, prompts: list[str], **kwargs: object) -> dict[str, torch.Tensor]:
        result = super().__call__(prompts, **kwargs)
        valid = result["attention_mask"].to(torch.bool)
        result["input_ids"][valid] = 7
        return result


def test_complete_prompt_tokenization_audit_roundtrips_and_records_full_tasks(
    tmp_path: Path,
) -> None:
    audit = audit_complete_prompt_tokenization(
        ("move the blue block", "open the drawer", "move the blue block"),
        _CharacterTokenizer(),
        maximum_tokens=128,
        use_qwen3_chat_template=True,
    )

    assert audit.prompt_count == 3
    assert tuple(entry.task for entry in audit.prompts) == (
        "move the blue block",
        "open the drawer",
    )
    assert audit.maximum_observed_tokens < audit.maximum_tokens
    assert audit.as_dict()["truncation_count"] == 0

    path = tmp_path / "prompt-audit.json"
    path.write_text(json.dumps(audit.as_dict()), encoding="utf-8")
    assert CompletePromptTokenizationAudit.load(path) == audit


def test_complete_prompt_tokenization_fails_before_silent_truncation() -> None:
    with pytest.raises(ContractError, match="would be truncated"):
        audit_complete_prompt_tokenization(
            ("move the blue block",),
            _CharacterTokenizer(),
            maximum_tokens=8,
            use_qwen3_chat_template=True,
        )


def test_complete_prompt_tokenization_fails_if_bounded_ids_change() -> None:
    with pytest.raises(ContractError, match="changed complete task semantics"):
        audit_complete_prompt_tokenization(
            ("open",),
            _MutatingBoundedTokenizer(),
            maximum_tokens=64,
            use_qwen3_chat_template=True,
        )


def test_complete_prompt_tokenization_rejects_tampered_report() -> None:
    audit = audit_complete_prompt_tokenization(
        ("open",),
        _CharacterTokenizer(),
        maximum_tokens=64,
        use_qwen3_chat_template=True,
    )
    payload = audit.as_dict()
    payload["maximum_tokens"] = 63

    with pytest.raises(ContractError, match="artifact digest differs"):
        CompletePromptTokenizationAudit.from_dict(payload)


def test_controlled_prompt_pairs_must_be_audited_and_token_distinct() -> None:
    audit = audit_complete_prompt_tokenization(
        ("open", "shut"),
        _CharacterTokenizer(),
        maximum_tokens=64,
        use_qwen3_chat_template=True,
    )
    validate_distinct_prompt_tokenizations(audit, (("open", "shut"),))

    with pytest.raises(ContractError, match="absent from the complete"):
        validate_distinct_prompt_tokenizations(audit, (("open", "missing"),))

    colliding = audit_complete_prompt_tokenization(
        ("open", "shut"),
        _CollidingTokenizer(),
        maximum_tokens=64,
        use_qwen3_chat_template=True,
    )
    with pytest.raises(ContractError, match="identical complete token IDs"):
        validate_distinct_prompt_tokenizations(colliding, (("open", "shut"),))


def test_lingbot_prompt_formatter_mirrors_both_upstream_paths() -> None:
    tokenizer = _CharacterTokenizer()
    assert (
        format_lingbot_prompt(
            "open",
            tokenizer,
            use_qwen3_chat_template=True,
        )
        == "<user>open</user>"
    )
    assert (
        format_lingbot_prompt(
            "open",
            tokenizer,
            use_qwen3_chat_template=False,
        )
        == "<bos>open\n"
    )
    assert (
        format_lingbot_prompt(
            "<bos>open\n",
            tokenizer,
            use_qwen3_chat_template=False,
        )
        == "<bos>open\n"
    )
