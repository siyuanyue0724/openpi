import os

os.environ["JAX_PLATFORMS"] = "cpu"

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi.models.pi0 import _TorchSonataRunner, _resolve_point_window_ids
import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)


def test_torch_sonata_runner_static_identity_is_value_based() -> None:
    runner_a = object.__new__(_TorchSonataRunner)
    runner_b = object.__new__(_TorchSonataRunner)
    runner_c = object.__new__(_TorchSonataRunner)

    for runner in (runner_a, runner_b, runner_c):
        runner._enc_out_dim = 512
        runner._in_channels = 6

    runner_a._cap = 1024
    runner_b._cap = 1024
    runner_c._cap = 2048

    runner_a._device = "cpu"
    runner_b._device = "cpu"
    runner_c._device = "cpu"

    class _InnerA:
        pass

    class _InnerB:
        pass

    runner_a._inner = _InnerA()
    runner_b._inner = _InnerA()
    runner_c._inner = _InnerB()

    assert runner_a == runner_b
    assert hash(runner_a) == hash(runner_b)
    assert runner_a != runner_c


def test_resolve_point_window_ids_defaults_to_tokenizer_tail() -> None:
    assert _resolve_point_window_ids(None, None, vocab_size=257152) == (257150, 257151)


def test_resolve_point_window_ids_rejects_mismatch() -> None:
    try:
        _resolve_point_window_ids(1, 2, vocab_size=257152)
    except RuntimeError as exc:
        assert "mismatch tokenizer" in str(exc)
    else:
        raise AssertionError("expected mismatch runtime error")


def test_embed_prefix_keeps_language_tokens_when_sonata_disabled() -> None:
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        enable_sonata=None,
    )
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)

    tokens, input_mask, ar_mask = model.embed_prefix(obs)

    image_token_count = 0
    for name in obs.images:
        image_tokens, _ = model.PaliGemma.img(obs.images[name], train=False)
        image_token_count += int(image_tokens.shape[1])
    text_emb = model.PaliGemma.llm(obs.tokenized_prompt, method="embed")

    assert tokens.shape[1] == image_token_count + int(text_emb.shape[1])
    assert input_mask.shape == (1, tokens.shape[1])
    assert ar_mask.shape == (tokens.shape[1],)
    assert jnp.all(input_mask)
