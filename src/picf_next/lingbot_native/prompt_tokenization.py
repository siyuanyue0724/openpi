"""Fail-closed complete-prompt tokenization audit for the LingBot host."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from picf_next.contracts import ContractError

COMPLETE_PROMPT_TOKENIZATION_SCHEMA = "picf-next.lingbot-complete-prompt-tokenization.v1"

_ENTRY_FIELDS = {
    "formatted_prompt_sha256",
    "task",
    "task_sha256",
    "token_count",
    "token_ids_sha256",
}
_AUDIT_FIELDS = {
    "artifact_sha256",
    "maximum_observed_tokens",
    "maximum_tokens",
    "prompt_count",
    "prompts",
    "schema",
    "truncation_count",
    "unique_prompt_count",
    "use_qwen3_chat_template",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _require_sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_task(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError("prompt task must be a non-empty string")
    return value


def _as_single_integer_sequence(value: object, name: str) -> tuple[int, ...]:
    dynamic_value: Any = value
    if (
        hasattr(dynamic_value, "detach")
        and hasattr(dynamic_value, "cpu")
        and hasattr(dynamic_value, "tolist")
    ):
        value = dynamic_value.detach().cpu().tolist()
    elif hasattr(dynamic_value, "tolist"):
        value = dynamic_value.tolist()
    if not isinstance(value, list):
        raise ContractError(f"{name} must be a batch-shaped integer sequence")
    if len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ContractError(f"{name} must contain only integers")
    return tuple(value)


def format_lingbot_prompt(
    task: str,
    tokenizer: Any,
    *,
    use_qwen3_chat_template: bool,
) -> str:
    """Mirror the released LingBot ``prepare_language`` prompt formatting."""

    task = _require_task(task)
    if not isinstance(use_qwen3_chat_template, bool):
        raise ContractError("Qwen3 chat-template selection must be boolean")
    if use_qwen3_chat_template:
        formatted = tokenizer.apply_chat_template(
            [{"role": "user", "content": task}],
            tokenize=False,
            add_generation_prompt=False,
        )
        if not isinstance(formatted, str) or not formatted:
            raise ContractError("Qwen3 chat template returned an invalid prompt")
        return formatted
    formatted = task if task.startswith("<bos>") else f"<bos>{task}"
    return formatted if formatted.endswith("\n") else f"{formatted}\n"


def _tokenize_prompt(
    tokenizer: Any,
    formatted: str,
    *,
    maximum_tokens: int | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    kwargs: dict[str, object] = {
        "return_tensors": "pt",
    }
    if maximum_tokens is None:
        kwargs.update(
            {
                "padding": False,
                "truncation": False,
            }
        )
    else:
        kwargs.update(
            {
                "padding": "max_length",
                "padding_side": "right",
                "max_length": maximum_tokens,
                "truncation": True,
            }
        )
    tokenized = tokenizer.__call__([formatted], **kwargs)
    if not isinstance(tokenized, Mapping):
        raise ContractError("LingBot tokenizer must return a mapping")
    if set(("input_ids", "attention_mask")) - set(tokenized):
        raise ContractError("LingBot tokenizer omitted IDs or attention mask")
    token_ids = _as_single_integer_sequence(tokenized["input_ids"], "prompt token IDs")
    attention_mask = _as_single_integer_sequence(
        tokenized["attention_mask"],
        "prompt attention mask",
    )
    if len(token_ids) != len(attention_mask) or not token_ids:
        raise ContractError("prompt token IDs and attention mask have incompatible lengths")
    if any(value not in (0, 1) for value in attention_mask):
        raise ContractError("prompt attention mask must be binary")
    return token_ids, attention_mask


@dataclass(frozen=True, slots=True)
class PromptTokenizationEntry:
    """One unique raw task and its exact complete token sequence identity."""

    task: str
    task_sha256: str
    formatted_prompt_sha256: str
    token_count: int
    token_ids_sha256: str

    def __post_init__(self) -> None:
        task = _require_task(self.task)
        if self.task_sha256 != _text_sha256(task):
            raise ContractError("prompt task digest does not match its UTF-8 text")
        _require_sha256(self.formatted_prompt_sha256, "formatted prompt digest")
        _require_positive_int(self.token_count, "prompt token count")
        _require_sha256(self.token_ids_sha256, "prompt token-ID digest")

    def as_dict(self) -> dict[str, object]:
        return {
            "formatted_prompt_sha256": self.formatted_prompt_sha256,
            "task": self.task,
            "task_sha256": self.task_sha256,
            "token_count": self.token_count,
            "token_ids_sha256": self.token_ids_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> PromptTokenizationEntry:
        if not isinstance(value, Mapping) or set(value) != _ENTRY_FIELDS:
            raise ContractError("prompt tokenization entry fields differ from schema")
        return cls(
            task=_require_task(value["task"]),
            task_sha256=_require_sha256(value["task_sha256"], "prompt task digest"),
            formatted_prompt_sha256=_require_sha256(
                value["formatted_prompt_sha256"],
                "formatted prompt digest",
            ),
            token_count=_require_positive_int(value["token_count"], "prompt token count"),
            token_ids_sha256=_require_sha256(
                value["token_ids_sha256"],
                "prompt token-ID digest",
            ),
        )


@dataclass(frozen=True, slots=True)
class CompletePromptTokenizationAudit:
    """Immutable proof that LingBot receives every token of every source task."""

    prompt_count: int
    maximum_tokens: int
    use_qwen3_chat_template: bool
    prompts: tuple[PromptTokenizationEntry, ...]

    def __post_init__(self) -> None:
        _require_positive_int(self.prompt_count, "source prompt count")
        _require_positive_int(self.maximum_tokens, "maximum prompt tokens")
        if not isinstance(self.use_qwen3_chat_template, bool):
            raise ContractError("Qwen3 chat-template selection must be boolean")
        if not self.prompts:
            raise ContractError("prompt tokenization audit requires at least one unique prompt")
        tasks = tuple(entry.task for entry in self.prompts)
        if tasks != tuple(sorted(set(tasks))):
            raise ContractError("prompt tokenization entries must have unique sorted tasks")
        if self.prompt_count < len(self.prompts):
            raise ContractError("source prompt count cannot be smaller than unique prompt count")
        if self.maximum_observed_tokens > self.maximum_tokens:
            raise ContractError("prompt tokenization audit contains a truncated prompt")

    @property
    def maximum_observed_tokens(self) -> int:
        return max(entry.token_count for entry in self.prompts)

    @property
    def artifact_sha256(self) -> str:
        return _sha256(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "maximum_observed_tokens": self.maximum_observed_tokens,
            "maximum_tokens": self.maximum_tokens,
            "prompt_count": self.prompt_count,
            "prompts": [entry.as_dict() for entry in self.prompts],
            "schema": COMPLETE_PROMPT_TOKENIZATION_SCHEMA,
            "truncation_count": 0,
            "unique_prompt_count": len(self.prompts),
            "use_qwen3_chat_template": self.use_qwen3_chat_template,
        }

    def as_dict(self) -> dict[str, object]:
        return {
            **self._payload(),
            "artifact_sha256": self.artifact_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> CompletePromptTokenizationAudit:
        if not isinstance(value, Mapping) or set(value) != _AUDIT_FIELDS:
            raise ContractError("complete prompt tokenization fields differ from schema")
        if value["schema"] != COMPLETE_PROMPT_TOKENIZATION_SCHEMA:
            raise ContractError("complete prompt tokenization schema differs")
        if _require_nonnegative_int(value["truncation_count"], "truncation count") != 0:
            raise ContractError("complete prompt tokenization report records truncation")
        raw_prompts = value["prompts"]
        if not isinstance(raw_prompts, list):
            raise ContractError("complete prompt tokenization prompts must be a list")
        result = cls(
            prompt_count=_require_positive_int(value["prompt_count"], "source prompt count"),
            maximum_tokens=_require_positive_int(value["maximum_tokens"], "maximum prompt tokens"),
            use_qwen3_chat_template=value["use_qwen3_chat_template"],
            prompts=tuple(PromptTokenizationEntry.from_dict(item) for item in raw_prompts),
        )
        if value["unique_prompt_count"] != len(result.prompts):
            raise ContractError("complete prompt tokenization unique count differs")
        if value["maximum_observed_tokens"] != result.maximum_observed_tokens:
            raise ContractError("complete prompt tokenization maximum length differs")
        if _require_sha256(value["artifact_sha256"], "prompt audit digest") != (
            result.artifact_sha256
        ):
            raise ContractError("complete prompt tokenization artifact digest differs")
        return result

    @classmethod
    def load(cls, path: Path) -> CompletePromptTokenizationAudit:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def audit_complete_prompt_tokenization(
    tasks: Sequence[str],
    tokenizer: Any,
    *,
    maximum_tokens: int,
    use_qwen3_chat_template: bool,
) -> CompletePromptTokenizationAudit:
    """Prove that the released bounded tokenizer preserves every source token."""

    maximum_tokens = _require_positive_int(maximum_tokens, "maximum prompt tokens")
    if not isinstance(tasks, Sequence) or isinstance(tasks, str | bytes) or not tasks:
        raise ContractError("prompt tokenization audit requires a non-empty task sequence")
    normalized_tasks = tuple(_require_task(task) for task in tasks)
    entries: list[PromptTokenizationEntry] = []
    for task in sorted(set(normalized_tasks)):
        formatted = format_lingbot_prompt(
            task,
            tokenizer,
            use_qwen3_chat_template=use_qwen3_chat_template,
        )
        complete_ids, complete_mask = _tokenize_prompt(
            tokenizer,
            formatted,
            maximum_tokens=None,
        )
        if any(value != 1 for value in complete_mask):
            raise ContractError("unbounded prompt tokenization unexpectedly emitted padding")
        bounded_ids, bounded_mask = _tokenize_prompt(
            tokenizer,
            formatted,
            maximum_tokens=maximum_tokens,
        )
        retained_ids = tuple(
            token_id
            for token_id, valid in zip(bounded_ids, bounded_mask, strict=True)
            if valid == 1
        )
        if len(complete_ids) > maximum_tokens:
            raise ContractError(
                f"LingBot prompt would be truncated ({len(complete_ids)} > "
                f"{maximum_tokens} tokens): {task!r}"
            )
        if retained_ids != complete_ids:
            raise ContractError(
                f"LingBot bounded tokenizer changed complete task semantics: {task!r}"
            )
        entries.append(
            PromptTokenizationEntry(
                task=task,
                task_sha256=_text_sha256(task),
                formatted_prompt_sha256=_text_sha256(formatted),
                token_count=len(complete_ids),
                token_ids_sha256=_sha256(list(complete_ids)),
            )
        )
    return CompletePromptTokenizationAudit(
        prompt_count=len(normalized_tasks),
        maximum_tokens=maximum_tokens,
        use_qwen3_chat_template=use_qwen3_chat_template,
        prompts=tuple(entries),
    )


def validate_distinct_prompt_tokenizations(
    audit: CompletePromptTokenizationAudit,
    prompt_pairs: Sequence[tuple[str, str]],
) -> None:
    """Require every controlled prompt pair to reach distinct complete token IDs."""

    if not isinstance(audit, CompletePromptTokenizationAudit):
        raise TypeError("prompt-pair validation requires a complete tokenization audit")
    if (
        not isinstance(prompt_pairs, Sequence)
        or isinstance(prompt_pairs, str | bytes)
        or not prompt_pairs
    ):
        raise ContractError("prompt-pair validation requires at least one pair")
    digest_by_task = {entry.task: entry.token_ids_sha256 for entry in audit.prompts}
    for pair_index, pair in enumerate(prompt_pairs):
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or any(not isinstance(task, str) or not task for task in pair)
        ):
            raise ContractError(f"prompt pair {pair_index} must contain two non-empty tasks")
        missing = tuple(task for task in pair if task not in digest_by_task)
        if missing:
            raise ContractError(
                f"prompt pair {pair_index} is absent from the complete tokenization audit: "
                f"{missing!r}"
            )
        if digest_by_task[pair[0]] == digest_by_task[pair[1]]:
            raise ContractError(
                f"prompt pair {pair_index} collapses to identical complete token IDs"
            )
