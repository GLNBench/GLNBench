"""Focused tests for label-noise generation and validation."""

import numpy as np
import pytest
import torch

from util.noise import deterministic, noise_operation


def test_deterministic_noise_corrupts_exact_compact_subset_count():
    """Compact train+val labels must use compact, not global, positions."""
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    positions = torch.arange(labels.numel())

    noisy, noisy_indices = noise_operation(
        labels,
        features=None,
        n_classes=3,
        noise_type="deterministic",
        noise_rate=0.5,
        noise_seed=7,
        idx_train=positions,
        debug=False,
    )

    assert len(noisy_indices) == 3
    assert torch.count_nonzero(noisy != labels).item() == 3
    assert np.all(noisy_indices < labels.numel())


def test_deterministic_noise_is_reproducible():
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    positions = torch.arange(labels.numel())

    first, _, _ = deterministic(labels, positions, noise_rate=0.5, seed=11)
    second, _, _ = deterministic(labels, positions, noise_rate=0.5, seed=11)

    np.testing.assert_array_equal(first, second)


def test_deterministic_noise_rejects_global_indices_for_compact_labels():
    labels = torch.tensor([0, 1, 2])
    global_node_ids = torch.tensor([4, 9, 12])

    with pytest.raises(ValueError, match="outside labels"):
        deterministic(labels, global_node_ids, noise_rate=0.5, seed=1)


@pytest.mark.parametrize("rate", [-0.01, 1.01])
def test_noise_rate_must_be_a_probability(rate):
    with pytest.raises(ValueError, match="between 0 and 1"):
        noise_operation(
            torch.tensor([0, 1]), None, 2,
            noise_type="uniform", noise_rate=rate, debug=False,
        )


def test_corruption_rejects_single_class_labels():
    with pytest.raises(ValueError, match="at least two classes"):
        noise_operation(
            torch.zeros(4, dtype=torch.long), None, 1,
            noise_type="uniform", noise_rate=0.2, debug=False,
        )
