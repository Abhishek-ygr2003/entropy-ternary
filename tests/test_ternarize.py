"""
Unit tests for ternarization utilities.
Run with: pytest tests/
"""

import math
import torch
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model      import MLP
from src.ternarize  import (
    ternarize_weight_row,
    select_random,
    select_l2_norm,
    select_entropy_guided,
    compress_fc1,
    compress_fc2_only,
    simulate_int8,
    op_count,
    op_count_fc2_only,
    _ternary_entropy,
)


# ─────────────────────────────────────────────────────────────
# ternarize_weight_row
# ─────────────────────────────────────────────────────────────

class TestTernarizeWeightRow:
    def test_zero_output_for_near_zero_weights(self):
        w = torch.zeros(10)
        out = ternarize_weight_row(w)
        assert out.abs().max().item() == 0.0

    def test_output_shape_preserved(self):
        w = torch.randn(128)
        out = ternarize_weight_row(w)
        assert out.shape == w.shape

    def test_effective_weight_nonzero_for_large_weights(self):
        w = torch.tensor([2.0, -2.0, 0.01])
        out = ternarize_weight_row(w)
        # First two should be non-zero, last should be zero
        assert out[0].item() != 0.0
        assert out[1].item() != 0.0
        assert out[2].item() == 0.0

    def test_clipped_magnitude_range(self):
        w = torch.randn(64)
        out = ternarize_weight_row(w)
        nonzero = out[out != 0]
        if len(nonzero) > 0:
            assert (nonzero.abs() >= 0.1).all()
            assert (nonzero.abs() <= 0.9).all()


# ─────────────────────────────────────────────────────────────
# Selection strategies
# ─────────────────────────────────────────────────────────────

class TestSelectionStrategies:
    def setup_method(self):
        self.weight = torch.randn(128, 784)
        self.frac   = 0.50
        self.n_sel  = math.floor(0.50 * 128)  # 64

    def test_random_returns_correct_count(self):
        idx = select_random(self.weight, self.frac, seed=0)
        assert len(idx) == self.n_sel

    def test_random_deterministic_with_same_seed(self):
        idx1 = select_random(self.weight, self.frac, seed=99)
        idx2 = select_random(self.weight, self.frac, seed=99)
        assert idx1 == idx2

    def test_random_different_seeds_differ(self):
        idx1 = select_random(self.weight, self.frac, seed=0)
        idx2 = select_random(self.weight, self.frac, seed=1)
        assert idx1 != idx2

    def test_l2_returns_correct_count(self):
        idx = select_l2_norm(self.weight, self.frac)
        assert len(idx) == self.n_sel

    def test_l2_deterministic(self):
        idx1 = select_l2_norm(self.weight, self.frac)
        idx2 = select_l2_norm(self.weight, self.frac)
        assert idx1 == idx2

    def test_entropy_returns_correct_count(self):
        idx = select_entropy_guided(self.weight, self.frac)
        assert len(idx) == self.n_sel

    def test_entropy_deterministic(self):
        idx1 = select_entropy_guided(self.weight, self.frac)
        idx2 = select_entropy_guided(self.weight, self.frac)
        assert idx1 == idx2

    def test_all_indices_within_range(self):
        for fn in [
            lambda: select_random(self.weight, self.frac, seed=42),
            lambda: select_l2_norm(self.weight, self.frac),
            lambda: select_entropy_guided(self.weight, self.frac),
        ]:
            for idx in fn():
                assert 0 <= idx < 128


# ─────────────────────────────────────────────────────────────
# Entropy
# ─────────────────────────────────────────────────────────────

class TestEntropy:
    def test_entropy_non_negative(self):
        w = torch.randn(784)
        h = _ternary_entropy(w)
        assert h >= 0.0

    def test_entropy_upper_bound(self):
        # Maximum entropy for 3-symbol distribution is log2(3) ≈ 1.585
        w = torch.randn(784)
        h = _ternary_entropy(w)
        assert h <= math.log2(3) + 1e-6

    def test_entropy_zero_for_uniform_sign(self):
        # All weights far above threshold → all +1 → entropy 0
        w = torch.ones(784) * 10.0
        h = _ternary_entropy(w)
        assert h < 0.01


# ─────────────────────────────────────────────────────────────
# Model compression
# ─────────────────────────────────────────────────────────────

class TestCompression:
    def setup_method(self):
        torch.manual_seed(0)
        self.model = MLP()

    def test_compress_fc1_does_not_modify_original(self):
        w_before = self.model.fc1.weight.data.clone()
        compress_fc1(self.model, 0.5, "entropy_guided")
        assert torch.allclose(self.model.fc1.weight.data, w_before)

    def test_compress_fc1_all_strategies_return_model(self):
        for strat in ["random", "l2_norm", "entropy_guided"]:
            comp = compress_fc1(self.model, 0.5, strat, seed=42)
            assert isinstance(comp, MLP)

    def test_compress_fc2_does_not_modify_original(self):
        w_before = self.model.fc2.weight.data.clone()
        compress_fc2_only(self.model)
        assert torch.allclose(self.model.fc2.weight.data, w_before)

    def test_int8_preserves_shape(self):
        m8 = simulate_int8(self.model)
        assert m8.fc1.weight.shape == self.model.fc1.weight.shape
        assert m8.fc2.weight.shape == self.model.fc2.weight.shape


# ─────────────────────────────────────────────────────────────
# Operation count
# ─────────────────────────────────────────────────────────────

class TestOpCount:
    def setup_method(self):
        torch.manual_seed(0)
        self.model = MLP()

    def test_op_reduction_positive(self):
        result = op_count(self.model, 0.5, "entropy_guided")
        assert result["op_reduction"] > 0.0

    def test_op_reduction_less_than_100(self):
        result = op_count(self.model, 0.5, "entropy_guided")
        assert result["op_reduction"] < 100.0

    def test_higher_fraction_generally_higher_reduction(self):
        r30 = op_count(self.model, 0.3, "entropy_guided")
        r50 = op_count(self.model, 0.5, "entropy_guided")
        # More neurons ternarized → more zero-skipping → higher reduction
        assert r50["op_reduction"] >= r30["op_reduction"]

    def test_fp32_ops_matches_architecture(self):
        result = op_count(self.model, 0.5, "random", seed=0)
        # 784*128 + 128*10 = 100352 + 1280 = 101632
        assert result["fp32_ops"] == 784 * 128 + 128 * 10


# ─────────────────────────────────────────────────────────────
# Hard determinism test for entropy-guided selection
# ─────────────────────────────────────────────────────────────

class TestEntropyGuidedDeterminism:
    """Verify that entropy-guided compression is bit-identical across runs."""

    def setup_method(self):
        torch.manual_seed(0)
        self.model = MLP()

    def test_entropy_guided_weights_identical_across_runs(self):
        """Run entropy-guided compression twice on the same checkpoint
        and assert torch.allclose on the resulting FC1 weights."""
        comp1 = compress_fc1(self.model, 0.5, "entropy_guided")
        comp2 = compress_fc1(self.model, 0.5, "entropy_guided")
        assert torch.allclose(comp1.fc1.weight.data, comp2.fc1.weight.data), \
            "Entropy-guided compression produced different weights across two runs"

    def test_entropy_guided_selection_identical_across_runs(self):
        """Verify the selected neuron indices are identical across runs."""
        weight = self.model.fc1.weight.data
        idx1 = select_entropy_guided(weight, 0.5)
        idx2 = select_entropy_guided(weight, 0.5)
        assert idx1 == idx2, \
            "Entropy-guided selection produced different indices across two runs"

