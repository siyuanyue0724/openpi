import logging
import re

import numpy as np
import sentencepiece
from transformers import AutoProcessor
from typing import List

import openpi.shared.download as download

# SpatialLM 窗口标记（作为“用户自定义 token”使用）
POINT_START = "<|point_start|>"
POINT_END   = "<|point_end|>"
__all__ = ["PaligemmaTokenizer", "FASTTokenizer",
           "POINT_START", "POINT_END", "FAST_POINT_START_ID", "FAST_POINT_END_ID"]

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

    # ---------- 小工具：FAST←→PaliGemma 的数值映射（严格互逆） ----------
    def _act_tokens_to_paligemma_tokens(self, tokens: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(tokens, list):
            tokens = np.array(tokens)
        return self._paligemma_tokenizer.vocab_size() - 1 - self._fast_skip_tokens - tokens

    def _normalize_prompt_segment(self, s: str) -> str:
        # 只对“非特殊片段”做轻度清洗，以保持和历史实现一致：
        #  - 转小写
        #  - 把 '_' 当作空格
        return s.lower().replace("_", " ")

    # ---- 对“提示词片段”做注入：遇到窗口标记 → 直接放入固定 id；其余片段 → 正常 SPM 编码 ----
    # 与 SpatialLM 的 user-defined token 语义逐 token 等价。
    def _encode_prompt_with_specials(self, prompt_text: str, *, add_bos_first_segment: bool) -> list[int]:
        if not isinstance(prompt_text, str):
            prompt_text = str(prompt_text)
        parts = re.split(rf'({re.escape(POINT_START)}|{re.escape(POINT_END)})', prompt_text)
        out: list[int] = []
        # 仅当要求时才在“本段”的开头加 BOS；通常我们会把 BOS 放在固定前缀 "Task: " 段里
        if add_bos_first_segment:
            bos_id = int(self._paligemma_tokenizer.bos_id())
            if bos_id < 0:
                raise RuntimeError("SentencePiece BOS id is invalid.")
            out.append(bos_id)
        for part in parts:
            if not part:
                continue
            if part == POINT_START:
                out.append(self._point_start_id)
            elif part == POINT_END:
                out.append(self._point_end_id)
            else:
                # 只清洗“普通片段”，保留窗口标记原样
                norm = self._normalize_prompt_segment(part)
                enc = self._paligemma_tokenizer.encode(norm, add_bos=False)
                out.extend(enc)
        return out

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # ！！！关键修复点：不要在整串上替换 '_'，否则破坏 <|point_start|>/<|point_end|>
        # 我们把窗口注入限制在“用户 prompt”片段里，并保持固定前后缀大小写与历史一致。
        prompt_raw = str(prompt).strip()

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # 拼装为三个段落，确保：
        #  (1) "Task: " 段负责加 BOS（大小写不变）
        #  (2) prompt 段做窗口注入 + 轻清洗（不加 BOS）
        #  (3) ", State: ...;\\n" 段正常编码（大小写不变）
        state_str = " ".join(map(str, discretized_state))
        prefix_tokens: list[int] = []
        prefix_tokens += self._paligemma_tokenizer.encode("Task: ", add_bos=True)
        prefix_tokens += self._encode_prompt_with_specials(prompt_raw, add_bos_first_segment=False)
        prefix_tokens += self._paligemma_tokenizer.encode(f", State: {state_str};\n", add_bos=False)

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

    # ---------- 新增：PG→FAST 逆映射 + 子序列查找（避免 decode/encode 不可逆） ----------
    def _pg_tokens_to_act_tokens(self, pg_tokens: np.ndarray | List[int]) -> np.ndarray:
        if isinstance(pg_tokens, list):
            pg_tokens = np.array(pg_tokens)
        vsz = int(self._paligemma_tokenizer.vocab_size())
        fast = vsz - 1 - self._fast_skip_tokens - pg_tokens
        return fast.astype(np.int32)

    @staticmethod
    def _find_subseq(arr: List[int], pat: List[int]) -> int:
        """Return start index of the first occurrence of pat in arr, or -1 if not found."""
        if not pat:
            return -1
        L, M = len(arr), len(pat)
        # 朴素搜索足够；序列很短
        for i in range(L - M + 1):
            if arr[i:i+M] == pat:
                return i
        return -1

    def extract_actions(self, tokens: np.ndarray, action_horizon: int, action_dim: int) -> np.ndarray:
        """严格逐 token 可逆：在 PG id 序列中定位动作段，并做数值逆映射得到 FAST ids。"""
        toks = tokens.tolist() if isinstance(tokens, np.ndarray) else list(tokens)
        # 在 id 序列中直接查找 "Action: " 与 "|" 两个锚点
        act_prefix_ids = self._paligemma_tokenizer.encode("Action: ")
        bar_ids        = self._paligemma_tokenizer.encode("|", add_eos=False)  # 只匹配 '|' 本体

        i = self._find_subseq(toks, act_prefix_ids)
        if i < 0:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        s = i + len(act_prefix_ids)

        j_rel = self._find_subseq(toks[s:], bar_ids)
        if j_rel < 0:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)
        e = s + j_rel

        if e <= s:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        pg_action_tokens = np.array(toks[s:e], dtype=np.int32)
        fast_tokens = self._pg_tokens_to_act_tokens(pg_action_tokens)
        # 过滤负值（万一预测出界）
        fast_tokens = fast_tokens[fast_tokens >= 0]
        if fast_tokens.size == 0:
            return np.zeros((action_horizon, action_dim), dtype=np.float32)

        return self._fast_tokenizer.decode(
            [fast_tokens.tolist()],
            time_horizon=action_horizon,
            action_dim=action_dim,
        )[0]

# ---------------- 模块级：暴露固定 special id，供 config 直接 import ----------------
def _paligemma_vocab_size() -> int:
    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        spp = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    return int(spp.vocab_size())
def _paligemma_bos_eos() -> tuple[int, int]:
    path = download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"})
    with path.open("rb") as f:
        spp = sentencepiece.SentencePieceProcessor(model_proto=f.read())
    return int(spp.bos_id()), int(spp.eos_id())

_PG_VSZ = _paligemma_vocab_size()
# 保留区末尾两个 id 作为窗口标记；与 FAST 动作 token（至多用到 vsz-129）不冲突
FAST_POINT_START_ID: int = _PG_VSZ - 2
FAST_POINT_END_ID  : int = _PG_VSZ - 1
_BOS_ID, _EOS_ID = _paligemma_bos_eos()
assert FAST_POINT_START_ID not in (_BOS_ID, _EOS_ID), "POINT_START id collides with BOS/EOS!"
assert FAST_POINT_END_ID   not in (_BOS_ID, _EOS_ID), "POINT_END id collides with BOS/EOS!"
assert FAST_POINT_START_ID < FAST_POINT_END_ID, "POINT ids order must be ascending."
assert FAST_POINT_START_ID >= _PG_VSZ - 128 and FAST_POINT_END_ID >= _PG_VSZ - 128, \
    "window token ids must stay in the top-128 reserved range to avoid collision with FAST tokens."

logging.info(
    "[tokenizer] FAST_POINT_START_ID=%d, FAST_POINT_END_ID=%d (pg_vocab=%d)",
    FAST_POINT_START_ID, FAST_POINT_END_ID, _PG_VSZ
)