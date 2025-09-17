import jax
import jax.numpy as jnp

from ginseng.nn import (
    nn_xavier_uniform,
    nn_init_linear,
    nn_linear,
    nn_dropout,
    nn_normalize,
    nn_annotate_init,
    nn_annotate,
    nn_annotate_loss,
)


def test_nn_xavier_uniform():
    key = jax.random.key(123)
    W = nn_xavier_uniform(key, (10, 5))
    assert W.shape == (10, 5)


def test_nn_init_linear():
    key = jax.random.key(123)
    params = nn_init_linear(key, 10, 5)
    assert params["W"].shape == (10, 5)
    assert params["b"].shape == (5,)


def test_nn_linear():
    key = jax.random.key(123)
    x = jax.random.uniform(key, (5, 10))
    params = nn_init_linear(key, 10, 5)
    z = nn_linear(params, x)
    assert z.shape == (5, 5)


def test_nn_dropout():
    key = jax.random.key(123)
    x = jnp.ones((10, 10))
    z_drop = nn_dropout(key, x, rate=0.5, training=True)
    z_keep = nn_dropout(key, x, rate=0.5, training=False)
    assert (z_drop == 0).sum() > 0
    assert (z_keep == 0).sum() == 0


def test_nn_normalize():
    x = jnp.ones((10, 5))
    z = nn_normalize(x, target_sum=1.0)
    assert (z == jnp.log1p(1.0 / 5.0)).all()


def test_nn_annotate_init():
    key = jax.random.key(123)

    params = nn_annotate_init(key, n_genes=10, n_classes=2, hidden_dim=32)

    assert params["features1"]["W"].shape == (10, 10)
    assert params["features2"]["W"].shape == (10, 10)
    assert params["attn1"]["W"].shape == (10, 32)
    assert params["attn2"]["W"].shape == (32, 10)
    assert params["head"]["W"].shape == (10, 2)


def test_nn_annotate():
    key = jax.random.key(123)

    params = nn_annotate_init(key, n_genes=5, n_classes=2, hidden_dim=32)

    x = jax.random.poisson(key, 3, (10, 5))

    logits = nn_annotate(
        params,
        key,
        x,
        dropout_rate=0.5,
        normalize=True,
        target_sum=None,
        return_attn=False,
        training=True,
    )

    assert logits.shape == (10, 2)


def test_nn_annotate_loss():
    key = jax.random.key(123)
    x = jax.random.uniform(key, (100, 10))
    y = jax.random.randint(key, (100,), minval=0, maxval=4)

    params = nn_annotate_init(key, n_genes=10, n_classes=2, hidden_dim=32)

    loss = nn_annotate_loss(params, key, x, y)
    assert loss > 0
