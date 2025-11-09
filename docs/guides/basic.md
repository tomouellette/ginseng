# Basic usage

## `data`

`ginseng` uses `zarr` stores for efficient storage and retrieval of single-cell count data. This makes it easy to cache, split, and iterate over data for training models.

### Preparing a `GinsengDataset`

We can create a `GinsengDataset` by providing a list of gene names or, alternatively, a column in `adata.obs` that specifies which genes to include in our dataset (i.e. a boolean mask). These genes will then be subset during the construction of a `GinsengDataset`. 

!!! warning
    `ginseng` expects raw, unnormalized counts, so make sure to double check when creating a `GinsengDataset`.


Below we provide an example of creating a `GinsengDataset` using randomly assigned genes.

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
```

`GinsengDataset` also lets you specify a `group` argument when creating the dataset. By setting `group`, you can ensure that cells from the same group are not split across the training and validation sets. For example, in the code below, we set `group="donor_id"`, which ensures that all cells from a given donor are kept entirely within either the training or validation split.

```python
# Create a GinsengDataset with additional group information for downstream splitting
GinsengDataset.create(
    "ginseng.zarr",
    adata,
	labels="cell_type",
	gene_key="gene_symbol"
    gene_mask=adata.var.include_genes,
	groups="donor_id",
)
```

Since a `GinsengDataset` is just a `zarr` array, the array's chunk size will dictate I/O speeds. By default, `GinsengDataset` will set the chunk size to (`256`, number of genes). However, if you know what batch size you will use during training, then setting chunk size to `batch_size` will lead to the fastest/most efficient reading as chunks won't overlap.

```python
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

!!! tip
    `GinsengDataset` is compatible with the backed mode of AnnData objects. This allows you to create large datasets efficiently by loading data from disk as needed, rather than reading the entire dataset into memory.

### Working with a `GinsengDataset`

Once a `GinsengDataset` has been created, it can be reloaded at any time while remaining memory efficient, since only the required chunks are read into memory. `GinsengDataset` also supports flexible batching, optional label balancing, and automatic train/holdout splitting.

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

## `train`

`ginseng` provides a simple interface for training a lightweight neural attention model for cell type annotation.

### Training a model with `GinsengTrainer`

Given a `GinsengDataset`, end-to-end training and logging for cell type annotation can be achieved using `GinsengTrainerSettings` and `GinsengTrainer`.

```python
from ginseng.train import GinsengTrainerSettings, GinsengTrainer

# Specify a variety of training settings if desired
settings = GinsengTrainerSettings(
	# Augmentation
	rate=0.1,
    lam_max=0.05,
    lower=0,
    upper=20,

    # Model
    hidden_dim=256,
    dropout_rate=0.25,

    # Optimization
    batch_size=128,
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,

    # Data
    normalize=True,
    target_sum=1e4,
    holdout_fraction=0.2,
    balance_train=True,
    group_level=False,
    group_mode="fraction",

    # Randomness
    seed=1
)

# Train a model
logger, model_state = GinsengTrainer(
	dataset,
	settings,
	epochs=10,
	silent=False
)
```

Of note, `ginseng` implements a variety of optional data augmentations during training. These include:

- Cell-wise dropout/masking (randomly setting individual counts to zero at rate `rate`)
- Random addition of Poisson counts ($\lambda \in$ [0, `lam_max`]) to simulate background RNA
- Gene-wise masking, in which up to `g` genes are randomly zeroed per batch to mimic sparsity or a reduced gene panel.

While these augmentations don't perfectly match the apparent statistical properties of sc/snRNA data, they add additional variation that helps the model generalize (e.g. in cases where genes in the training set may be missing during prediction, or in cases where the distribution of ambient/background RNAis different from the unaugmented training data).

### Saving and loading a model trained with `GinsengTrainer`

The ability to perform annotation on new datasets requires information on (1) which genes were used during training (and their order), (2) the normalization and transformations applied to the data, and (3) the model weights. As such, `ginseng` provides functions to serialize and deserialize all this "model state" information as necessary. (Note that the state is currently saved in a portable `hdf5` format, but this may change in the future to improve compression/reduce size.)

```python
from ginseng.io import save_ginseng_state, load_ginseng_state

# Save a trained ginseng model
save_ginseng_state(model_state, "ginseng.state.hdf5"))

# Load a trained ginseng model
model_state = load_ginseng_state(model_state, "ginseng.state.hdf5")
```

## `annotate`

### Annotating new data with a trained `ginseng` model

If you've trained a model, annotation is simple. Just load your model state and provide an `AnnData` object or path to count data, and there you go. For optimal speed, you can tune batch size, but in most cases annotation should be fast (even for hundreds of thousands of cells). For more details on additional arguments, check the docs/docstring.

```python
from ginseng.annotate import GinsengAnnotate

GinsengAnnotate(
    model_state,
    adata, # Or path to 10x matrix market data, .h5 file, or .h5ad file
    gene_key="gene_symbol",
    batch_size=256,
)

# Get cell type labels
cell_types = adata.obs["ginseng_cell_type"].values
```

If you want to just access the probability (softmax) for each cell type label per cell, you can instead use the `return_probs` argument.

```python
cell_probs = GinsengAnnotate(
    model_state,
    adata,
    gene_key="gene_symbol",
    batch_size=256,
    return_probs=True
)

# Get most probable cell type label
cell_probs["cell_type"] = cell_probs.idxmax(axis=1)
```


## `io`

### Reading count data

`ginseng` provides a variety of functions to read single-cell count data such as 10x matrix market format, 10x h5, or AnnData objects.

```python
from ginseng.io import read_10x_mtx, read_10x_h5, read_adata

# Load 10x matrix market data
adata = read_10x_mtx("counts/")

# Load 10x h5 data
adata = read_10x_h5("counts.h5")

# Load AnnData object (anndata package)
adata = read_10x_h5("counts.h5")

# Load AnnData object backed (anndata package)
adata = read_10x_h5("counts.h5", backed="r")
```
