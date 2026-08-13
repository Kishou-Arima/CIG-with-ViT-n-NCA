"""Regression coverage for the CuPy inference and Torch backprop rollouts."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import cupy as cp
import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "pure_math_gpu_nca.py"
SPEC = importlib.util.spec_from_file_location("pure_math_gpu_nca", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
nca = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nca)


class RolloutParityTest(unittest.TestCase):
    def test_cupy_and_torch_rollouts_match(self) -> None:
        nca.cp = cp
        nca.torch = torch
        weights_cp = nca.load_or_create_nca_weights()
        cond_cp = cp.linspace(-0.75, 0.75, nca.COND_DIM, dtype=cp.float32)
        weights_torch = nca.torch_from_weights(weights_cp, "cpu", [])
        cond_torch = torch.tensor(cp.asnumpy(cond_cp), dtype=torch.float32)

        rgb_cp, state_cp = nca.nca_rollout(cond_cp, weights_cp, 13, 11, 4, 0.5, 0.15)
        rgb_torch, state_torch = nca.torch_nca_rollout(cond_torch, weights_torch, 13, 11, 4, 0.5, 0.15)

        np.testing.assert_allclose(cp.asnumpy(rgb_cp), rgb_torch.detach().numpy(), rtol=2e-4, atol=2e-5)
        np.testing.assert_allclose(cp.asnumpy(state_cp), state_torch.detach().numpy(), rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
