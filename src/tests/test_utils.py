import jax.numpy as jnp
import numpy as np
import pytest

from anndata import AnnData

from ginseng.utils import iter_sequential, compute_hvgs


@pytest.fixture
def adata():
    np.random.seed(123)
    X = np.vstack(
        [
            np.random.normal(10.0, 3.0, size=100),
            np.random.normal(10.0, 3.0, size=100),
            np.random.normal(10.0, 3.0, size=100),
            np.random.normal(90.0, 1.0, size=100),
            np.random.normal(90.0, 1.0, size=100),
            np.random.normal(90.0, 1.0, size=100),
        ]
    )

    X[0:3, 0] = 10.0
    X[3:6, 0] = 90.0

    X[0:6, -5:] = np.array([1, 2, 3, 4, 5])

    return AnnData(
        X,
        var={
            "gene": np.arange(100).astype(str),
        },
        obs={"cell_type": 3 * [0] + 3 * [1], "donor": [*range(6)]},
    )


def test_iter_sequential(adata):
    for i, (X, y, batch_idx) in enumerate(
        iter_sequential(adata, "cell_type", batch_size=3)
    ):
        assert isinstance(
            X, jnp.ndarray
        ), "Failure: `X` not jnp.ndarray in test_iter_sequential."
        assert isinstance(
            X, jnp.ndarray
        ), "Failure: y not jnp.ndarray in test_iter_sequential."

        assert X.shape == (
            3,
            100,
        ), "Failure: Wrong batch_size in test_iter_sequential."

        if i == 0:
            assert (batch_idx == jnp.array([0, 1, 2])).all()
            assert (y == 0).sum() == 3
        else:
            assert (batch_idx == jnp.array([3, 4, 5])).all()
            assert (y == 1).sum() == 3


def test_iter_sequential_with_order(adata):
    gene_order = np.arange(100).astype(str)
    gene_order[-5] = "99"
    gene_order[-4] = "98"
    gene_order[-3] = "97"
    gene_order[-2] = "96"
    gene_order[-1] = "95"

    for i, (X, y, batch_idx) in enumerate(
        iter_sequential(adata, "cell_type", batch_size=1, gene_order=gene_order)
    ):
        assert X[0][-5] == 5
        assert X[0][-4] == 4
        assert X[0][-3] == 3
        assert X[0][-2] == 2
        assert X[0][-1] == 1


def test_compute_hvgs(adata):
    compute_hvgs(adata, n_top_genes=5)
    assert "ginseng_genes" in adata.var.columns
    assert adata.var.ginseng_genes.sum() == 5
