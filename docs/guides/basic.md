# Usage

## `GinsengDataset`

`GinsengDataset` is the core object for training single-cell annotation models in `ginseng`.
It is designed to convert large single-cell datasets (from `.h5ad`, 10x, or
URLs) into a compressed on-disk `zarr` store for training. This allows for
memory-efficient training by streaming mini-batches without loading the entire
dataset into RAM.

### Selecting highly variable genes

To reduce dimensionlity of the input data, `ginseng` provides built-in support for selecting highly variable genes from in-memory or backed AnnData. The `select_hvgs` function is implemented using
a chunk-based strategy so it will identify HVGs even for very large datasets without loading everything into memory.

```python
from ginseng.data.io import read_adata
from ginseng.utils import select_hvgs

# Load AnnData object
adata = read_adata("data.h5ad", backed="r")

# Select highly variable genes
select_hvgs(adata, n_top_genes=2000)

# Access highly variable genes
assert "ginseng_genes" in adata.var
```

### Creating a `GinsengDataset`

To convert your raw counts into a `GinsengDataset`, the `create` class method can be used. The `.create` method handles gene subsetting, label encoding, and disk serialization.

```python
from ginseng.data import GinsengDataset

# Create a dataset from an existing AnnData object or path
ds = GinsengDataset.create(
    path="my_dataset.zarr",          # Output directory
    adata="path/to/counts.h5ad",     # Input data (Path, URL, or AnnData)
    label_key="cell_type",           # Target labels in adata.obs
    layer="counts",                  # Optional: specify a layer (e.g., raw counts)
    genes="ginseng_genes",           # Optional: use a mask in adata.var
    group_key="batch"                # Optional: metadata for stratified splitting
)
```

### Loading an existing `GinsengDataset`

If you have already created a `GinsengDataset` and saved it to disk, you can load it directly by providing the path to the `zarr` store.

```python
# Load an existing dataset from disk
ds = GinsengDataset("dataset.zarr")
```

### Initializing train/test splits

Once the Zarr store is created, you can define your data splits. Ginseng stores
these indices inside the Zarr group so they persist across sessions.

```python
# Create a 90/10 train/test split
ds.make_split(fraction=0.1, stratify_group=True)
```

If you used `group_key` when creating the `GinsengDataset`, then setting
`stratify_group=True` will ensure that all cells from the same group instance
(e.g. batch, donor, etc.) are kept together in either the training or test set.

### Streaming mini-batches

The stream method provides a python iterator that yields jax/numpy compatible arrays. This is where the on-disk performance provides major benefits as only the required chunks are decompressed during iteration.

```python
# Stream mini-batches for training
for X_batch, y_batch in ds.stream(batch_size=256, split="train", shuffle=True, balance_labels=True):
    # X_batch shape: (256, n_genes)
    # y_batch shape: (256,)
    # Perform training step here...
    pass
```

The `stream` method supports shuffling, different splits (`train`, `test`, or
`all`), adjustable batch sizes, and can also enforce balanced sampling of
labels during training.

## `GinsengClassifier`

Training cell type annotation models in `ginseng` is managed by the
`GinsengClassifierTrainer`. This class handles the `jax`-based optimization
loop, handles on-the-fly data augmentation, and manages model evaluation on
holdout splits.

### Setting training parameters

A `GinsengClassifier` can be trained using a variety of model, optimization,
and augmentation hyperparameters. These can be set using the
`GinsengClassifierTrainerSettings` class.

```python
from ginseng.train import GinsengClassifierTrainerSettings

settings = GinsengClassifierTrainerSettings(
    # Augmentation parameters
    rate=0.1,
    lam_max=None,
    lower=0,
    upper=200,

    # Model parameters
    hidden_dim=64,
    dropout_rate=0.2,
    batch_size=128,

    # Training parameters
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
    normalize=True,
    target_sum=1e4,
    holdout_fraction=0.05,
    balance_train=True,
    group_level=False,
    group_mode="fraction",

    # Random seed for reproducibility
    seed=123
)
```

### Training a model

Once you've specified your dataset and training parameters, you can initialize a `GinsengClassifierTrainer` to begin training.

```python
from ginseng.train import GinsengClassifierTrainer

# Initialize trainer
trainer = GinsengClassifierTrainer(dataset, settings)

# Fit the model
model, state = trainer.fit(epochs=10)
```

After training, the `fit` method returns the trained model and its state, which can be used for inference or saved to disk.

### Saving and loading model state

The model state can be saved to disk for later use or inference.

```python
from ginseng.data.io import save_model, load_model

# Save model state
save_model(state, "model.h5")

# Load model state
loaded_state = load_model("model.h5")
```

## Annotating new data

Once you have trained a model, you can either directly use the model on new
data by iterating over the dataset. However, the recommended approach is to use
the `ginseng.classify` function, which uses the model state to correctly
subset, order, and account for missing genes. Furthermore, it's optimized to work
with backed AnnData objects which makes it easy to annotate large datasets without
having to load everything into memory.

```python
import ginseng
from ginseng.data.io import read_adata, load_model

# Load model
state = load_model("model.h5")

# Load AnnData
adata = read_adata("data.h5ad")

# Classify new data in-place
ginseng.classify(data, state, layer="counts")
assert "ginseng_cell_type" in adata.obs
assert "ginseng_confidence" in adata.obs

# Return predictions as a separate table
predictions = ginseng.classify(data, state, layer="counts", return_table=True)
```
