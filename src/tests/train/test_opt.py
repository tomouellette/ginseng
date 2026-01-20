# Copyright (c) 2026, Tom Ouellette
# Licensed under the MIT License

import jax
import jax.numpy as jnp
import pytest
from ginseng.train.opt import opt_init_adam, opt_adam_update


@pytest.fixture
def linear_params():
    """Initial weights for a simple linear model."""
    return {"beta": jnp.array([0.5]), "intercept": jnp.array([0.5])}


@pytest.fixture
def synthetic_data():
    """Simple y = x dataset."""
    x = jnp.linspace(-10, 10, 50).reshape(-1, 1)
    y = x  # Perfect linear relationship
    return x, y


def loss_fn(params, x, y):
    """Mean squared error loss."""
    yhat = params["beta"] * x + params["intercept"]
    return jnp.mean((y - yhat) ** 2)


class TestAdamOptimizer:
    """Test suite for Adam optimizer implementation."""

    def test_opt_init_adam(self, linear_params):
        """Verify state initialization creates zeroed moments and step 0."""
        state = opt_init_adam(linear_params)

        assert state.step == 0
        assert jnp.all(state.m["beta"] == 0)
        assert jnp.all(state.v["intercept"] == 0)
        assert state.m["beta"].shape == linear_params["beta"].shape

    def test_opt_adam_convergence(self, linear_params, synthetic_data):
        """Verify that the optimizer can solve a trivial linear regression."""
        x, y = synthetic_data
        params = linear_params
        state = opt_init_adam(params)

        @jax.jit
        def trainstep(p, s, x_batch, y_batch):
            loss, grads = jax.value_and_grad(loss_fn)(p, x_batch, y_batch)
            new_params, new_state = opt_adam_update(
                grads, p, s, lr=1e-1, weight_decay=0.0
            )
            return new_params, new_state, loss

        # Track initial loss
        _, _, initial_loss = trainstep(params, state, x, y)

        # Optimization loop
        current_loss = initial_loss
        for _ in range(200):
            params, state, current_loss = trainstep(params, state, x, y)

        # Assertions
        assert current_loss < initial_loss
        assert current_loss < 1e-3, f"Loss {current_loss} too high for trivial task"

        # Verify parameters converged toward beta=1.0, intercept=0.0
        assert jnp.isclose(params["beta"], 1.0, atol=1e-2)
        assert jnp.isclose(params["intercept"], 0.0, atol=1e-2)

    def test_weight_decay_influence(self, linear_params):
        """Verify that weight decay actually changes the update direction."""
        params = linear_params
        state = opt_init_adam(params)

        # Create a dummy gradient of 0
        grads = jax.tree.map(jnp.zeros_like, params)

        # Update without weight decay (should do nothing since grads are 0)
        p_no_wd, _ = opt_adam_update(grads, params, state, weight_decay=0.0)
        # Update with heavy weight decay
        p_wd, _ = opt_adam_update(grads, params, state, lr=0.1, weight_decay=1.0)

        # Without WD, params should stay at 0.5
        assert jnp.allclose(p_no_wd["beta"], 0.5)
        # With WD and 0 grad, params should be pulled toward zero (0.5 - 0.1 * (1.0 * 0.5) = 0.45)
        assert p_wd["beta"] < 0.5
