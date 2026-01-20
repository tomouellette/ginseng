# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ginseng.model.nn import (
    nn_xavier_uniform,
    nn_init_linear,
    nn_linear,
    nn_dropout,
    nn_normalize,
    nn_annotate_init,
    nn_annotate,
    nn_annotate_loss,
    GinsengClassifier,
)


@pytest.fixture
def rng_key():
    """Base PRNG key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def model_dims():
    """Standard dimensions for testing."""
    return {"n_genes": 10, "n_classes": 3, "hidden_dim": 16, "batch_size": 4}


@pytest.fixture
def sample_data(model_dims):
    """Synthetic expression data and labels."""
    rng = np.random.default_rng(42)
    x = rng.integers(
        0, 100, size=(model_dims["batch_size"], model_dims["n_genes"])
    ).astype(np.float32)
    y = rng.integers(0, model_dims["n_classes"], size=(model_dims["batch_size"],))
    return jnp.array(x), jnp.array(y)


class TestNN:
    """Test low-level neural building blocks."""

    def test_xavier_uniform_shape(self, rng_key):
        shape = (10, 20)
        weights = nn_xavier_uniform(rng_key, shape)
        assert weights.shape == shape
        # Check if roughly within Xavier limits: sqrt(6 / (10 + 20)) approx 0.447
        assert jnp.all(weights >= -0.5) and jnp.all(weights <= 0.5)

    def test_nn_init_linear(self, rng_key):
        params = nn_init_linear(rng_key, 10, 5)
        assert params["W"].shape == (10, 5)
        assert params["b"].shape == (5,)
        assert jnp.all(params["b"] == 0)

    def test_nn_linear_forward(self):
        params = {"W": jnp.ones((2, 3)), "b": jnp.array([1.0, 2.0, 3.0])}
        x = jnp.ones((1, 2))  # [1, 1]
        # (1*1 + 1*1) + bias = [2+1, 2+2, 2+3]
        out = nn_linear(params, x)
        expected = jnp.array([[3.0, 4.0, 5.0]])
        assert jnp.allclose(out, expected)

    def test_nn_dropout_eval_mode(self, rng_key):
        x = jnp.ones((10, 10))
        # In eval mode, dropout should be identity
        out = nn_dropout(rng_key, x, rate=0.5, training=False)
        assert jnp.array_equal(x, out)

    def test_nn_dropout_zero_rate(self, rng_key):
        x = jnp.ones((10, 10))
        out = nn_dropout(rng_key, x, rate=0.0, training=True)
        assert jnp.array_equal(x, out)


class TestDataProcessing:
    """Test normalization and transformation logic."""

    def test_nn_normalize(self):
        # Create data where first row sums to 10 and second to 20
        x = jnp.array([[5.0, 5.0], [10.0, 10.0]])
        target = 100.0
        out = nn_normalize(x, target_sum=target)

        # Row 1: log1p(100 * (5/10)) = log1p(50)
        # Row 2: log1p(100 * (10/20)) = log1p(50)
        expected_val = jnp.log1p(50.0)
        assert jnp.allclose(out, expected_val)
        assert out.shape == (2, 2)


class TestModelFunctions:
    """Test the full annotation model logic."""

    def test_nn_annotate_shapes(self, rng_key, model_dims, sample_data):
        x, _ = sample_data
        params = nn_annotate_init(
            rng_key,
            model_dims["n_genes"],
            model_dims["n_classes"],
            model_dims["hidden_dim"],
        )

        logits = nn_annotate(params, rng_key, x, training=False)
        assert logits.shape == (model_dims["batch_size"], model_dims["n_classes"])

    def test_nn_annotate_with_attention(self, rng_key, model_dims, sample_data):
        x, _ = sample_data
        params = nn_annotate_init(
            rng_key, model_dims["n_genes"], model_dims["n_classes"]
        )

        logits, attn = nn_annotate(params, rng_key, x, return_attn=True, training=False)
        # Attention should have same shape as input features (gated attention)
        assert attn.shape == x.shape
        assert jnp.all(attn >= 0) and jnp.all(attn <= 1)

    def test_nn_annotate_loss(self, rng_key, model_dims, sample_data):
        x, y = sample_data
        params = nn_annotate_init(
            rng_key, model_dims["n_genes"], model_dims["n_classes"]
        )

        loss = nn_annotate_loss(params, rng_key, x, y)
        assert loss.shape == ()  # Scalar
        assert loss > 0


class TestGinsengClassifier:
    """Test the Object-Oriented wrapper."""

    def test_classifier_init(self, model_dims):
        clf = GinsengClassifier(
            n_genes=model_dims["n_genes"], n_classes=model_dims["n_classes"]
        )
        assert "head" in clf.params
        assert clf.params["head"]["W"].shape == (
            model_dims["n_genes"],
            model_dims["n_classes"],
        )

    def test_classifier_key_management(self, model_dims):
        clf = GinsengClassifier(model_dims["n_genes"], model_dims["n_classes"])
        key1 = clf._get_key()
        key2 = clf._get_key()
        assert not jnp.array_equal(key1, key2)

    def test_classifier_predict(self, model_dims, sample_data):
        x, _ = sample_data
        clf = GinsengClassifier(model_dims["n_genes"], model_dims["n_classes"])

        logits = clf.predict(x)
        assert logits.shape == (model_dims["batch_size"], model_dims["n_classes"])

    def test_classifier_evaluate(self, model_dims, sample_data):
        x, y = sample_data
        clf = GinsengClassifier(model_dims["n_genes"], model_dims["n_classes"])

        loss, acc = clf.evaluate(x, y)
        assert isinstance(loss, float)
        assert 0.0 <= acc <= 1.0
