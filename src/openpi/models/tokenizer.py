import logging
import re

import numpy as np
import sentencepiece
from transformers import AutoProcessor

import openpi.shared.download as download

# SpatialLM 窗口标记（作为“用户自定义 token”使用）
POINT_START = "<|point_start|>"
POINT_END   = "<|point_end|>"

class PaligemmaTokenizer:
    def __init__(self, max_len: int = 48):
        self._max_len = max_len

        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())

    def tokenize(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        cleaned_text = prompt.strip().replace("_", " ").replace("\n", " ")
        # tokenize "\n" separately as the "start of answer" token
        tokens = self._tokenizer.encode(cleaned_text, add_bos=True) + self._tokenizer.encode("\n")
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            mask = [True] * tokens_len + padding
            tokens = tokens + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            mask = [True] * self._max_len

        return np.asarray(tokens), np.asarray(mask)


class FASTTokenizer:
    def __init__(self, max_len: int = 256, fast_tokenizer_path: str = "physical-intelligence/fast"):
        self._max_len = max_len

        # Download base PaliGemma tokenizer
        path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
        with path.open("rb") as f:
            self._paligemma_tokenizer = sentencepiece.SentencePieceProcessor(model_proto=f.read())
        self._pg_vocab_size = int(self._paligemma_tokenizer.vocab_size())

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        # Skip last 128 tokens in PaliGemma vocab since they are special / reserved
        self._fast_skip_tokens = 128

        # SpatialLM-style window fence: 末尾两个 id
        # （与动作 token 映射区间严格不冲突）
        self._point_start_id = FAST_POINT_START_ID
        self._point_end_id   = FAST_POINT_END_ID
        self._special_token_to_id = {
            POINT_START: self._point_start_id,
            POINT_END:   self._point_end_id,
        }
        logging.info(
            "[FASTTokenizer] spatial window token ids: %s=%d, %s=%d",
            POINT_START, self._point_start_id, POINT_END, self._point_end_id
        )

    # ---- 提供最小 id 查询接口，供 config 等模块安全获取 ----
    def token_to_id(self, tok: str) -> int | None:
        return self._special_token_to_id.get(tok, None)

    def convert_tokens_to_ids(self, toks: list[str]) -> list[int | None]:
        return [self.token_to_id(t) for t in toks]

    # ---- “整串编码 + 特殊标记原位注入”，等价于 SP 的 user-defined token 语义 ----
    def _encode_text_with_specials(self, text: str, *, add_bos_first_segment: bool) -> list[int]:
        if not isinstance(text, str):
            text = str(text)
        parts = re.split(rf'({re.escape(POINT_START)}|{re.escape(POINT_END)})', text)
        out: list[int] = []
        bos_done = False
        for part in parts:
            if not part:
                continue
            if part == POINT_START:
                out.append(self._point_start_id)
            elif part == POINT_END:
                out.append(self._point_end_id)
            else:
                enc = self._paligemma_tokenizer.encode(part, add_bos=add_bos_first_segment and not bos_done)
                if add_bos_first_segment and not bos_done and enc:
                    bos_done = True
                out.extend(enc)
        return out

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # 前缀整串编码（若含窗口标记，则在该位置注入单一 id；否则与旧实现完全等价）
        state_str = " ".join(map(str, discretized_state))
        prefix_str = f"Task: {cleaned_text}, State: {state_str};\n"
        prefix_tokens = self._encode_text_with_specials(prefix_str, add_bos_first_segment=True)

        if actions is not None:
            # Tokenize actions with FAST tokenizer --> map to last tokens in PaliGemma vocab
            action_tokens = self._fast_tokenizer(actions[None])[0]
            action_tokens_in_pg = self._act_tokens_to_paligemma_tokens(action_tokens)

            # Convention: postfix contains 'Action:' followed by FAST tokens, followed by '|'
            postfix_tokens = (
                self._paligemma_tokenizer.encode("Action: ")
                + action_tokens_in_pg.tolist()
                + self._paligemma_tokenizer.encode("|", add_eos=True)
            )
        else:
            postfix_tokens = []

        # Create output token sequence & masks
        # AR mask is 0 on prefix (bidirectional attention) and 1 on postfix (causal attention to all previous tokens)
        tokens = prefix_tokens + postfix_tokens
        token_mask = [True] * len(tokens)
        ar_mask = [0] * len(prefix_tokens) + [1] * len(postfix_tokens)
        loss_mask = [False] * len(prefix_tokens) + [True] * len(postfix_tokens)  # Loss on postfix only

        # Pad tokens to max length
        tokens_len = len(tokens)
        if tokens_len < self._max_len:
            padding = [False] * (self._max_len - tokens_len)
            tokens = tokens + padding
            token_mask = token_mask + padding
            ar_mask = ar_mask + padding
            loss_mask = loss_mask + padding
        else:
            if len(tokens) > self._max_len:
                logging.warning(
                    f"Token length ({len(tokens)}) exceeds max length ({self._max_len}), truncating. "
                    "Consider increasing the `max_token_len` in your model config if this happens frequently."
                )
            tokens = tokens[: self._max_len]
            token_mask = token_mask[: self._max_len]
            ar_mask = ar_mask[: self._max_len]
            loss_mask = loss_mask[: self._max_len]

        return np.asarray(tokens), np.asarray(token_mask), np.asarray(ar_mask), np.asarray(loss_mask)

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        # Decode predicted output tokens
        decoded_tokens = self._paligemma_tokenizer.decode(tokens.tolist())

        # Extract actions from FAST model outputs
        if "Action: " not in decoded_tokens:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        # Extract actions from decoded tokens
        raw_action_tokens = np.array(
            self._paligemma_tokenizer.encode(decoded_tokens.split("Action: ")[1].split("|")[0].strip())
        )
        action_tokens = self._act_tokens_to_paligemma_tokens(raw_action_tokens)
        return self._fast_tokenizer.decode(
            [action_tokens.tolist()], time_horizon=action_horizon, action_dim=action_dim
        )[0]

    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens

# ---------------- 模块级：暴露固定 special id，供 config 直接 import ----------------
def _paligemma_vocab_size() -> int:
    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        spp = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    return int(spp.vocab_size())

_PG_VSZ = _paligemma_vocab_size()
# 保留区末尾两个 id 作为窗口标记；与 FAST 动作 token（至多用到 vsz-129）不冲突
FAST_POINT_START_ID: int = _PG_VSZ - 2
FAST_POINT_END_ID  : int = _PG_VSZ - 1
logging.info(
    "[tokenizer] FAST_POINT_START_ID=%d, FAST_POINT_END_ID=%d (pg_vocab=%d)",
    FAST_POINT_START_ID, FAST_POINT_END_ID, _PG_VSZ
)