# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import jax
import jax.numpy as jnp
import pytest

from ginseng.train.augment import (
    augment,
    augment_mask,
    augment_background,
    augment_dropgene,
)


@pytest.fixture
def rng_key():
    """Provide a fresh PRNG key."""
    return jax.random.key(123)


@pytest.fixture
def sample_matrix():
    """Provide a standard 10x10 ones matrix for masking/dropping."""
    return jnp.ones((10, 10))


class TestAugmentation:
    """Test suite for scRNA-seq data augmentation functions."""

    def test_augment_mask(self, rng_key, sample_matrix):
        """Verify that masking zeros out elements based on the rate."""
        # Rate 0 should do nothing
        no_mask = augment_mask(rng_key, sample_matrix, rate=0.0)
        assert jnp.allclose(no_mask, sample_matrix)
        assert no_mask.sum() == 100

        # Rate 0.5 should mask roughly half
        masked = augment_mask(rng_key, sample_matrix, rate=0.5)
        assert masked.sum() < 100
        assert jnp.any(masked == 0)

    def test_augment_background(self, rng_key):
        """Verify Poisson background noise addition."""
        x_zeros = jnp.zeros((10, 10))

        # lam_max 0 should add nothing
        no_noise = augment_background(rng_key, x_zeros, lam_max=0.0)
        assert jnp.all(no_noise == 0)

        # High lam_max should add significant noise
        noisy = augment_background(rng_key, x_zeros, lam_max=10.0)
        assert noisy.sum() > 0
        assert jnp.all(noisy >= 0)

    def test_augment_dropgene(self, rng_key):
        """Verify whole-gene dropping within lower/upper bounds."""
        n_cells, n_genes = 10, 5
        x = jnp.ones((n_cells, n_genes))

        # Drop 0 genes
        no_drop = augment_dropgene(rng_key, x, lower=0, upper=0)
        assert no_drop.sum() == n_cells * n_genes

        # Drop all genes (lower=5, upper=6 means exactly 5 genes dropped)
        all_drop = augment_dropgene(rng_key, x, lower=5, upper=6)
        assert jnp.all(all_drop == 0)

        # Drop exactly 1 gene (10 cells * 1 gene = 10 elements removed)
        one_drop = augment_dropgene(rng_key, x, lower=1, upper=2)
        assert one_drop.sum() == (n_cells * (n_genes - 1))

    def test_augment_orchestrator(self, rng_key, sample_matrix):
        """Verify the main augment function dispatches all steps."""
        # Test with all augmentations enabled
        out = augment(rng_key, sample_matrix, rate=0.1, lam_max=1.0, lower=1, upper=2)

        assert out.shape == sample_matrix.shape
        # With background noise + drops, the sum will be different,
        # but the shape and finiteness must be preserved.
        assert jnp.all(jnp.isfinite(out))

    def test_augment_no_op(self, rng_key, sample_matrix):
        """Verify augment returns original data if all params are None/0."""
        out = augment(
            rng_key, sample_matrix, rate=None, lam_max=None, lower=None, upper=None
        )
        assert jnp.allclose(out, sample_matrix)
