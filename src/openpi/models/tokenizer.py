import logging
import re

import numpy as np
import sentencepiece
from transformers import AutoProcessor

import openpi.shared.download as download

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

        # Instantiate FAST tokenizer
        self._fast_tokenizer = AutoProcessor.from_pretrained(fast_tokenizer_path, trust_remote_code=True)
        self._fast_skip_tokens = 128  # Skip last 128 tokens in PaliGemma vocab since they are special tokens
        # Skip last 128 tokens in PaliGemma vocab for FAST/special controls
        self._fast_skip_tokens = 128

        # ------------------------------ special ids for SpatialLM window ------------------------------
        # We allocate two ids from the reserved tail of the Paligemma vocab.  Action tokens are mapped into
        # [0, vocab_size - 1 - _fast_skip_tokens] via _act_tokens_to_paligemma_tokens, so these last two are free.
        self._pg_vocab_size = int(self._paligemma_tokenizer.vocab_size())
        # keep order stable: start < end
        self._point_start_id = self._pg_vocab_size - 2
        self._point_end_id   = self._pg_vocab_size - 1
        self._special_token_to_id = {
            POINT_START: self._point_start_id,
            POINT_END:   self._point_end_id,
        }
        logging.info(
            "[FASTTokenizer] spatial window token ids: %s=%d, %s=%d",
            POINT_START, self._point_start_id, POINT_END, self._point_end_id
        )

    # ---- tiny public helpers so training/config.py can fetch ids without .encode() ----
    def token_to_id(self, tok: str) -> int | None:
        """Return id only for known special tokens; otherwise None."""
        return self._special_token_to_id.get(tok, None)

    def convert_tokens_to_ids(self, toks: list[str]) -> list[int | None]:
        """Batch version used by some callers; only handles our two specials."""
        return [self.token_to_id(t) for t in toks]

    # --------------------------------- internal utils ---------------------------------
    def _encode_text_with_specials(self, text: str, *, add_bos_first_segment: bool) -> list[int]:
        """
        Encode plain text, but inject POINT_START / POINT_END as dedicated single ids.
        SentencePiece handles normal segments; markers are inserted verbatim as our special ids.
        """
        if not isinstance(text, str):
            text = str(text)
        # split and keep delimiters
        parts = re.split(rf'({re.escape(POINT_START)}|{re.escape(POINT_END)})', text)
        out: list[int] = []
        first_plain_done = False
        for part in parts:
            if part == "" or part is None:
                continue
            if part == POINT_START:
                out.append(self._point_start_id)
            elif part == POINT_END:
                out.append(self._point_end_id)
            else:
                # normal text → sentencepiece; BOS only once if requested
                enc = self._paligemma_tokenizer.encode(
                    part,
                    add_bos=add_bos_first_segment and not first_plain_done
                )
                if add_bos_first_segment and not first_plain_done and len(enc) > 0:
                    first_plain_done = True
                out.extend(enc)
        return out

    def tokenize(
        self, prompt: str, state: np.ndarray, actions: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cleaned_text = prompt.lower().strip().replace("_", " ")

        # Convention: state gets discretized into 256 discrete bins (assumed range after normalization: [-1, 1])
        discretized_state = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1

        # Convention: prefix includes prompt and string-representation of state, followed by ';'
        state_str = " ".join(map(str, discretized_state))
        # Build prefix tokens with in-place special markers:
        #   [BOS] + "Task: " + cleaned_text(with specials) + ", State: {state_str};\n"
        prefix_tokens  = self._paligemma_tokenizer.encode("Task: ", add_bos=True)
        prefix_tokens += self._encode_text_with_specials(cleaned_text, add_bos_first_segment=False)
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
