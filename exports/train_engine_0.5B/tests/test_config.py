"""Tests for training_engine_tensor.config — architecture constants."""

from __future__ import annotations

from training_engine_tensor.engine_config import EngineConfig, set_global_config


def test_no_micro_batch_size_export():
    """MICRO_BATCH_SIZE and GLOBAL_BATCH_SIZE should NOT be in __all__."""
    set_global_config(EngineConfig())
    from training_engine_tensor import config

    assert "MICRO_BATCH_SIZE" not in config.__all__
    assert "GLOBAL_BATCH_SIZE" not in config.__all__


def test_model_constants_loaded():
    """Core architecture constants must load from model_spec.toml."""
    set_global_config(EngineConfig())
    from training_engine_tensor import config

    assert config.NUM_LAYERS == 24
    assert config.HIDDEN_SIZE == 1024
    assert config.NUM_HEADS == 16
    assert config.NUM_KV_HEADS == 2
    assert config.FFN_HIDDEN_SIZE == 4096
    assert config._VOCAB_SIZE == 73448
    assert config.MAX_SEQ_LENGTH == 4096


def test_no_micro_batch_size_attr():
    """After dead-constant removal, the module should not have these attrs."""
    set_global_config(EngineConfig())
    from training_engine_tensor import config

    assert not hasattr(config, "MICRO_BATCH_SIZE")
    assert not hasattr(config, "GLOBAL_BATCH_SIZE")


def test_internal_flop_constants_not_public():
    """FLOP constants are internal implementation details — not in __all__."""
    set_global_config(EngineConfig())
    from training_engine_tensor import config

    for name in (
        "FORWARD_FLOPS_PER_TOKEN",
        "STANDARD_FORWARD_FLOPS_PER_TOKEN",
        "TRAINING_FLOPS_PER_TOKEN",
        "STANDARD_TRAINING_FLOPS_PER_TOKEN",
    ):
        assert name not in config.__all__, f"{name} should not be in __all__"
        assert not hasattr(config, name), (
            f"{name} should be prefixed with underscore (internal constant)"
        )


def test_compute_training_flops_still_works():
    """compute_training_flops must still work after renaming internal constants."""
    set_global_config(EngineConfig())
    from training_engine_tensor import config

    flops = config.compute_training_flops(1)
    assert flops > 0
    flops_std = config.compute_training_flops_standard(1)
    assert flops_std > 0


def _expected_mtp_extra_per_token(layer_flops_per_token: int) -> int:
    """Per-MTP-layer extra forward FLOPs per token.

    Matches the closed-form physical-GEMM enumeration in
    ``harness/ref/reference/train_pure_mup_mtp.py`` for the EAGLE-style
    MTP block: one full transformer block + one extra LM-head call
    (tied weight but independent compute) + one eh_proj linear
    (2H -> H, no bias) of ``2 * (2H) * H = 4 H^2`` FLOPs/token.
    """
    from training_engine_tensor import config

    return (
        layer_flops_per_token
        + 2 * config.HIDDEN_SIZE * config._VOCAB_SIZE
        + 4 * config.HIDDEN_SIZE * config.HIDDEN_SIZE
    )


def test_compute_training_flops_no_mtp_matches_main_only():
    """MTP=0: flops must equal main-model-only training FLOPs (regression guard)."""
    set_global_config(EngineConfig(mtp_num_layers=0))
    from training_engine_tensor import config

    tokens = 4096
    main_fwd_std = (
        config.NUM_LAYERS * config._STANDARD_LAYER_FLOPS_PER_TOKEN
        + config._GLOBAL_FLOPS_PER_TOKEN
    )
    main_fwd_hw = (
        config.NUM_LAYERS * config._LAYER_FLOPS_PER_TOKEN
        + config._GLOBAL_FLOPS_PER_TOKEN
    )
    assert config.compute_training_flops_standard(tokens) == 3 * main_fwd_std * tokens
    assert config.compute_training_flops(tokens) == 3 * main_fwd_hw * tokens


def test_compute_training_flops_includes_mtp_layers():
    """MTP>=1: flops must include per-MTP-layer transformer + LM-head + eh_proj."""
    tokens = 4096
    for k in (1, 2):
        from training_engine_tensor.engine_config import _reset_global_config

        _reset_global_config()
        set_global_config(EngineConfig(mtp_num_layers=k))
        from training_engine_tensor import config

        main_fwd_std = (
            config.NUM_LAYERS * config._STANDARD_LAYER_FLOPS_PER_TOKEN
            + config._GLOBAL_FLOPS_PER_TOKEN
        )
        main_fwd_hw = (
            config.NUM_LAYERS * config._LAYER_FLOPS_PER_TOKEN
            + config._GLOBAL_FLOPS_PER_TOKEN
        )
        mtp_extra_std = k * _expected_mtp_extra_per_token(
            config._STANDARD_LAYER_FLOPS_PER_TOKEN
        )
        mtp_extra_hw = k * _expected_mtp_extra_per_token(config._LAYER_FLOPS_PER_TOKEN)

        assert config.compute_training_flops_standard(tokens) == (
            3 * (main_fwd_std + mtp_extra_std) * tokens
        )
        assert config.compute_training_flops(tokens) == (
            3 * (main_fwd_hw + mtp_extra_hw) * tokens
        )


def test_compute_training_flops_standard_one_mtp_layer_ratio():
    """Sanity-check MTP=1 / MTP=0 ratio against the known ~1.18 from prior analysis.

    For MiniCPM4-0.5B + 1 EAGLE layer, one MTP layer's training FLOPs
    are ~18% of the main-model training FLOPs (the 44.1% * 1.18 ~= 52%
    discrepancy that surfaced the MFU under-count). This guards against
    geometry-spec drift that would silently break the calibration.
    """
    tokens = 4096
    from training_engine_tensor.engine_config import _reset_global_config

    _reset_global_config()
    set_global_config(EngineConfig(mtp_num_layers=0))
    from training_engine_tensor import config

    no_mtp = config.compute_training_flops_standard(tokens)

    _reset_global_config()
    set_global_config(EngineConfig(mtp_num_layers=1))
    one_mtp = config.compute_training_flops_standard(tokens)

    ratio = one_mtp / no_mtp
    assert 1.15 < ratio < 1.22, f"unexpected MTP=1 / MTP=0 ratio: {ratio:.4f}"


def test_vocab_size_not_public():
    """VOCAB_SIZE should be an internal constant, not in __all__."""
    set_global_config(EngineConfig())
    from training_engine_tensor import config

    assert "VOCAB_SIZE" not in config.__all__


def test_effective_vocab_size_uses_engine_config():
    """get_effective_vocab_size() must read from EngineConfig."""
    set_global_config(EngineConfig(vocab_size=99999))
    from training_engine_tensor import config

    assert config.get_effective_vocab_size() == 99999
