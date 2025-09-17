# Usage

## Preparing datasets for training

`ginseng` uses `zarr` stores for efficient storage and retrieval of single-cell count data.
This makes it easy to cache, split, and iterate over data for training models.

```python
import numpy as np
from ginseng.io import read_adata
from ginseng.data import GinsengDataset

# Load your single-cell count data
adata = read_adata("adata.h5ad")

# Define which genes to include (boolean mask in adata.var)
adata.var["include_genes"] = np.random.randint(0, 2, size=len(adata.var)).astype(bool)

# Create a GinsengDataset and save it as a zarr store
GinsengDataset.create(
    "ginseng.zarr",
    adata,
    labels="cell_type",
	gene_key="gene_symbol"
    gene_mask=adata.var.include_genes,
)

# Create a GinsengDataset with additional group information for downstream splitting
GinsengDataset.create(
    "ginseng.zarr",
    adata,
	labels="cell_type",
	gene_key="gene_symbol"
    gene_mask=adata.var.include_genes,
	groups="donor_id",
)

# If you know your batch size in advance, manually set chunk size to optimize throughput
GinsengDataset.create(
    "ginseng.zarr",
    adata,
	labels="cell_type",
	gene_key="gene_symbol"
    gene_mask=adata.var.include_genes,
	chunk_size=128 # Or, (128, sum(adata.var.include_genes))
)
```

## Working with datasets

Once a dataset has been created, it can be reloaded at any time. `GinsengDataset` supports flexible batching, optional label balancing, and automatic train/holdout splits.

```python
import numpy as np
from ginseng.data import GinsengDataset

rng = np.random.default_rng(123)

# Load the cached dataset from disk
dataset = GinsengDataset("ginseng.zarr")

# Iterate through all samples in batches
for x, y in dataset.iter_batches(batch_size=1, shuffle=False, rng=rng):
    _ = x * x  # your model logic here

# Create a holdout split (e.g., for validation)
dataset.make_holdout(holdout_fraction=0.2)

# Iterate through the training set
for x, y in dataset.iter_batches(batch_size=1, shuffle=True, split="train", rng=rng):
    _ = x * x

# Iterate through the training set with label balancing
for x, y in dataset.iter_batches(
    batch_size=1,
    shuffle=True,
    split="train",
    balance_train=True,
	rng=rng
):
    _ = x * x

# Iterate through the holdout set
for x, y in dataset.iter_batches(batch_size=1, shuffle=False, split="holdout", rng=rng):
    _ = x * x
	
# If `groups` were provided, force unique groups not to appear in both train and holdout
dataset.make_holdout(holdout_fraction=0.2, group_level=True)

# If `groups` were provided, leave only one group out in the holdout
dataset.make_holdout(holdout_fraction=0.2, group_level=True, group_mode="loo")
```
