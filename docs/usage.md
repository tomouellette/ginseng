# Usage

- [`data`](#data)
  - [Preparing a `GinsengDataset`](#preparing-a-ginsengdataset)
  - [Working with a `GinsengDataset`](#working-with-a-ginsengdataset)
- [`train`](#train)
  - [Training a model with `GinsengTrainer`](#training-a-model-with-ginsengtrainer)
  - [Saving and loading a model trained with `GinsengTrainer`](#saving-and-loading-a-model-trained-with-ginsengtrainer)
- [`annotate](#annotate)
  - [Annotating new data with a trained `ginseng` model](#annotating-new-data-with-a-trained-ginseng-model)

## `data`

### Preparing a `GinsengDataset`

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

### Working with a `GinsengDataset`

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

## `train`

### Training a model with `GinsengTrainer`

`GinsengTrainer` enables end-to-end model training and logging given a `GinsengDataset`.

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

If you've trained a model, annotation is simple. Just load your model state and provide an `AnnData` object or path to count data, and there you go. For optimal speed, you can tune batch size, but in most cases annotation should be fast even for hundreds of thousands of cells. Check out the docstring for details on additional arguments.

```python
from ginseng.annotate import GinsengAnnotate

GinsengAnnotate(
    model_state,
    adata, # Or path to 10x matrix market data, .h5 file, or .h5ad file
    gene_key="gene_symbol",
    backed=True,
    batch_size=256,
)

cell_types = adata.obs["ginseng_cell_type"].values
```
