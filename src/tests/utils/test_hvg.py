import numpy as np
import pytest

from anndata import AnnData

from ginseng.utils import select_hvgs


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


def test_select_hvgs(adata):
    select_hvgs(adata, n_top_genes=5)
    assert "ginseng_genes" in adata.var.columns
    assert adata.var.ginseng_genes.sum() == 5
