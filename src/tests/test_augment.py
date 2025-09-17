import jax
import jax.numpy as jnp

from ginseng.augment import augment_mask, augment_background, augment_dropgene


def test_augment_mask():
    key = jax.random.key(123)
    x = jnp.ones((10, 10))
    a = augment_mask(key, x, rate=0.0)
    assert a.sum() == 100

    b = augment_mask(key, x, rate=0.5)
    assert b.sum() < 100


def test_augment_background():
    key = jax.random.key(123)
    x = jnp.zeros((10, 10))
    a = augment_background(key, x, lam_max=0.0)
    assert a.sum() == 0

    b = augment_background(key, x, lam_max=10.0)
    assert b.sum() > 0


def test_augment_dropgene():
    key = jax.random.key(123)
    x = jnp.ones((10, 5))
    a = augment_dropgene(key, x, lower=0, upper=0)
    assert a.sum() == 50

    b = augment_dropgene(key, x, lower=5, upper=6)
    assert b.sum() == 0

    c = augment_dropgene(key, x, lower=1, upper=2)
    assert c.sum() == 40
