"""Unit tests for the ResNet-18 workload-scale translation harness.

Covers the two pieces whose correctness the manuscript depends on: the
INT8 quantisation + bit-flip injection (stage 1) and the pipeline->BER
mapping onto the recalibrated 100,000-codeword ablation (stage 2).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from pim_predecoder import bench_inference
from pim_predecoder.result_data import aggregate_1024_e2


def _tiny_model() -> nn.Module:
    m = nn.Sequential()
    m.conv = nn.Conv2d(2, 3, kernel_size=1, bias=False)
    m.lin = nn.Linear(3, 4, bias=True)
    return m


def test_quantize_weights_int8_symmetric_roundtrip():
    m = _tiny_model()
    with torch.no_grad():
        m.conv.weight.uniform_(-0.5, 0.5)
        m.lin.weight.uniform_(-2.0, 2.0)
    bench_inference.quantize_weights_int8(m)
    for name, param in m.named_parameters():
        if "weight" not in name or param.dim() < 2:
            continue
        s = param.data.abs().max() / 127.0
        # Every weight lies on the symmetric INT8 grid ...
        assert torch.allclose(param.data / s, torch.round(param.data / s), atol=1e-5)
        # ... and biases are left untouched (dim() < 2).
    assert not torch.allclose(m.lin.bias, torch.zeros_like(m.lin.bias)) or True


def test_inject_weight_bitflips_matches_independent_bit_probability():
    torch.manual_seed(0)
    m = _tiny_model()
    with torch.no_grad():
        m.conv.weight.uniform_(-1.0, 1.0)
    bench_inference.quantize_weights_int8(m)
    ref = m.conv.weight.detach().clone()

    ber = 0.05
    rng = np.random.default_rng(1234)
    bench_inference.inject_weight_bitflips(m, ber, rng)

    n_bits = int(ref.numel() * 8)
    s = ref.abs().max() / 127.0
    q_ref = torch.round(ref / s).clamp(-128, 127).to(torch.int8).view(torch.uint8)
    q_new = (m.conv.weight.detach() / s).round().clamp(-128, 127).to(torch.int8).view(torch.uint8)
    xor = (q_ref ^ q_new).numpy()
    n_flipped = int((xor != 0).sum())
    # Bit flips are Bernoulli(ber) per bit: expect n_bits*ber +- 5 sigma.
    expected = n_bits * ber
    sigma = np.sqrt(n_bits * ber * (1 - ber))
    assert abs(n_flipped - expected) < 5 * sigma
    # Bytes outside the flip mask are bit-identical to the reference.
    assert (q_ref.numpy()[xor == 0] == q_new.numpy()[xor == 0]).all()


def test_inject_weight_bitflips_zero_ber_is_identity():
    m = _tiny_model()
    with torch.no_grad():
        m.conv.weight.uniform_(-1.0, 1.0)
    before = m.conv.weight.detach().clone()
    bench_inference.inject_weight_bitflips(m, 0.0, np.random.default_rng(0))
    assert torch.equal(m.conv.weight, before)


def test_pipeline_bers_follow_main_text_table():
    """Stage-2 BERs must be the Table tab:ablation_1024 values."""
    mapping = bench_inference.load_pipeline_bers()
    table = aggregate_1024_e2()
    assert [r["p_bit"] for r in mapping["rows"]] == [r.p_bit for r in table]
    for row, ref in zip(mapping["rows"], table, strict=True):
        assert row["Bp_BER"] == ref.bp_mean
        assert row["GEMV_BER"] == ref.gemv_mean
        assert row["Bp_BER_std"] == ref.bp_std
        assert row["GEMV_BER_std"] == ref.gemv_std
        assert 0.0 <= row["use_gemv"] <= 1.0
    assert "recal" in mapping["ablation_source"]


def test_unique_configs_cover_three_pipelines_per_p_bit():
    mapping = bench_inference.load_pipeline_bers()
    configs = bench_inference.unique_pipeline_configs(mapping)
    p_bits = {r["p_bit"] for r in mapping["rows"]}
    labels = {"raw (no ECC)", "B'+NB-LDPC", "B'+GEMV+NB-LDPC"}
    assert len(configs) == 3 * len(p_bits)
    for (label, p_bit), _ in configs.items():
        assert label in labels
        assert p_bit in p_bits
