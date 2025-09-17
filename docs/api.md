# API

## Table of Contents

- [augment](#augment)
  - [augment_mask](#function-augment_mask)
  - [augment_background](#function-augment_background)
  - [augment_dropgene](#function-augment_dropgene)
  - [augment](#function-augment)
- [data](#data)
  - [GinsengDataset](#class-ginsengdataset)
    - [_create_from_numpy](#method-_create_from_numpy)
    - [_create_from_sparse](#method-_create_from_sparse)
    - [create](#method-create)
    - [_reopen_readonly](#method-_reopen_readonly)
    - [_load_metadata](#method-_load_metadata)
    - [has_holdout](#method-has_holdout)
    - [make_holdout](#method-make_holdout)
    - [get_batch](#method-get_batch)
    - [iter_batches](#method-iter_batches)
    - [__len__](#method-__len__)
    - [__getitem__](#method-__getitem__)
- [io](#io)
  - [read_10x_mtx](#function-read_10x_mtx)
  - [read_10x_h5](#function-read_10x_h5)
  - [read_adata](#function-read_adata)
  - [save_ginseng_state](#function-save_ginseng_state)
  - [load_ginseng_state](#function-load_ginseng_state)
- [nn](#nn)
  - [nn_xavier_uniform](#function-nn_xavier_uniform)
  - [nn_init_linear](#function-nn_init_linear)
  - [nn_linear](#function-nn_linear)
  - [nn_dropout](#function-nn_dropout)
  - [nn_normalize](#function-nn_normalize)
  - [nn_annotate_init](#function-nn_annotate_init)
  - [nn_annotate](#function-nn_annotate)
  - [nn_annotate_loss](#function-nn_annotate_loss)
  - [nn_annotate_evaluate](#function-nn_annotate_evaluate)
- [opt](#opt)
  - [AdamState](#class-adamstate)
  - [opt_init_adam](#function-opt_init_adam)
  - [opt_adam_update](#function-opt_adam_update)
- [train](#train)
  - [GinsengTrainerSettings](#class-ginsengtrainersettings)
  - [GinsengLogger](#class-ginsenglogger)
    - [update](#method-update)
  - [GinsengModelState](#class-ginsengmodelstate)
  - [GinsengTrainer](#function-ginsengtrainer)
- [utils](#utils)
  - [iter_sequential](#function-iter_sequential)
  - [compute_hvgs](#function-compute_hvgs)

## `augment`

### Function `augment_mask`

Randomly mask out counts across cells.

**Parameters:**
- **key** (`Array`): PRNG key array for dropout mask.
- **x** (`Array`): Input tensor.
- **rate** (`float`): Dropout probability.

**Returns:**
- `Array`: Tensor with dropout applied.

### Function `augment_background`

Randomly add Poisson-distributed background noise to counts.

**Parameters:**
- **key** (`Array`): PRNG key array for noise generation.
- **x** (`Array`): Non-normalized count matrix.
- **lam_max** (`float`): Maximum mean of Poisson distribution for sampling added noise.

**Returns:**
- `Array`: Counts with added Poisson noise.

### Function `augment_dropgene`

Randomly zero out entire genes.

**Parameters:**
- **key** (`Array`): PRNG key array for mask generation.
- **x** (`Array`): Count matrix.
- **lower** (`int`): Minimum number of genes to mask out.
- **upper** (`int`): Maximum (exclusive) number of genes to mask out.

**Returns:**
- `Array`: Filtered counts.

### Function `augment`

Apply a combination of single-cell RNA relevant augmentations.

**Parameters:**
- **key** (`Array`): PRNG key array for mask generation.
- **x** (`Array`): Count matrix.
- **rate** (`float`): Dropout probability.
- **lam_max** (`float`): Maximum mean of Poisson distribution for sampling added noise.
- **lower** (`int`): Minimum number of genes to mask out.
- **upper** (`int`): Maximum (exclusive) number of genes to mask out.

**Returns:**
- `Array`: Augmented counts.

## `data`

### Class `GinsengDataset`

A zarr-based dataset for efficient cell-level access.

**Attributes:**
- **path** (`Path`): Path to ginseng dataset zarr file.
- **store** (`LocalStore`): Initialized local zarr store.
- **root** (`zarr.group`): Zarr group containing structured ginseng dataset.

#### Method `_create_from_numpy`

Create cached dataset from a dense numpy array.

**Parameters:**
- **path** (`str | Path`): Path to store the new dataset.
- **X** (`np.ndarray`): Dense count matrix.
- **labels** (`np.ndarray`): Array of labels for each barcode.
- **chunk_size** (`tuple[int, int]`): Chunk size for storing counts.
- **groups** (`np.ndarray`): Group-level information used to stratify holdout data. If provided, no unique group will have data in both training and holdout when using `make_holdout`.
- **genes** (`np.ndarray`): Gene names arranged in order used for training.
- **column_mask** (`np.ndarray`): Boolean mask specifying which columns to include.

**Returns:**
- `GinsengDataset`: Instance of the created dataset.

#### Method `_create_from_sparse`

Create cached dataset from a sparse count matrix.

**Parameters:**
- **path** (`str | Path`): Path to store the new dataset.
- **X** (`scipy.sparse.spmatrix`): Sparse count matrix.
- **labels** (`np.ndarray`): Array of labels for each barcode.
- **chunk_size** (`tuple[int, int]`): Chunk size for storing counts.
- **groups** (`np.ndarray`): Group-level information used to stratify holdout data. If provided, no unique group will have data in both training and holdout when using `make_holdout`.
- **genes** (`np.ndarray`): Gene names arranged in order used for training.
- **column_mask** (`np.ndarray`): Boolean mask specifying which columns to include.

**Returns:**
- `GinsengDataset`: Instance of the created dataset.

#### Method `create`

Autodetect array type and create a dataset.

**Parameters:**
- **path** (`str | Path`): Path to store the new dataset.
- **adata** (`AnnData`): Annotated data storing count matrix.
- **labels** (`np.ndarray | str`): Array of labels for each barcode or key in AnnData.obs specifying labels.
- **chunk_size** (`int | tuple[int, int]`): Chunk size for storing counts. An integer specifies row-wise chunk size and a tuple of integers specifies row and column-wise chunk size. When an integer or no chunk size is provided, the column-wise chunk size is set to number of the number of included genes.
- **groups** (`np.ndarray`): Array of group-level information for each barcode or key in AnnData.obs that specifies groups. Group-level informatiion is used to stratify holdout data.
- **gene_key** (`str`): If provided, the column where gene names are stored. By default, the index of `adata.var` is assumed to store the gene names.
- **gene_mask** (`np.ndarray`): Boolean mask specifying which genes to include.

**Returns:**
- `GinsengDataset`: Instance of the created  dataset.

#### Method `_reopen_readonly`

Helper to reopen store in read-only mode

#### Method `_load_metadata`

Load metadata from zarr store.

#### Method `has_holdout`

Whether the dataset has a train/holdout split.

#### Method `make_holdout`

Create and persist a train/holdout split.

**Parameters:**
- **holdout_fraction** (`float`): Fraction of samples to allocate to holdout. The total number of samples held out per label is determined by min(N labels per label) * `holdout_fraction`. When group_level is True, then holdout_fraction specifies the fraction of unique groups to hold out.
- **rng** (`np.random.Generator, optional`): Random number generator for reproducibility.
- **group_level** (`bool`): If True, split data based on group-level information.
- **group_mode** (`str`): If 'fraction' then N groups x `holdout_fraction` groups will be held out. If 'loo', then a single group will be held out (leave one out).

#### Method `get_batch`

Retrieve a batch of count data for given cell indices.

**Parameters:**
- **indices** (`np.ndarray`): Array of cell indices to retrieve.

**Returns:**
- `np.ndarray`: Expression matrix subset of shape (len(indices), n_features).

#### Method `iter_batches`

Iterate through the dataset in batches.

**Parameters:**
- **batch_size** (`int`): Number of samples per batch.
- **shuffle** (`bool`): Whether to shuffle the dataset before batching.
- **rng** (`np.random.Generator`): Random number generator used if shuffling.
- **split** (`str`): Subset of data to iterate over ("all", "train", or "holdout").
- **balance_train** (`bool`): If True, then force training set to have an equal number of each label. Yields ------ tuple[np.ndarray, np.ndarray] Batches of expression data and labels.

#### Method `__len__`

Return the number of samples in the dataset.

**Parameters:**
- **split** (`str`): Subset of data to count ("all", "train", or "holdout").

**Returns:**
- `int`: Number of samples in the chosen subset.

#### Method `__getitem__`

Retrieve a single sample by index.

**Parameters:**
- **idx** (`int`): Position within the chosen split.
- **split** (`str`): Subset of data to retrieve from ("all", "train", or "holdout").

**Returns:**
- `tuple of (np.ndarray, int)`: Expression vector and corresponding label.

## `io`

### Function `read_10x_mtx`

Read 10x Genomics mtx format into an AnnData object.

**Parameters:**
- **path** (`str`): Path to directory containing matrix.mtx, barcodes.tsv, and genes.tsv/features.tsv.
- **var_names** (`str`): Select 'gene_symbols' or 'gene_ids' as index for var.
- **make_unique** (`bool`): If True, make var_names unique.

**Returns:**
- `AnnData`: 

### Function `read_10x_h5`

Read 10x Genomics h5 format into an AnnData object.

**Parameters:**
- **path** (`str`): Path to 10x Genomics .h5 file.
- **genome** (`str, optional`): Genome to extract (if file contains multiple genomes). If None, will use the first genome available.
- **var_names** (`str`): Select 'gene_symbols' or 'gene_ids' as index for var.
- **make_unique** (`bool`): If True, make var_names unique.

**Returns:**
- `AnnData`: 

### Function `read_adata`

Read an AnnData object from various supported file formats.

**Parameters:**
- **path** (`str | Path`): Path to the input count data stored in 10x .h5 format, AnnData .h5ad format, or in a 10x matrix market format folder.
- **backed** (`bool`): If True and the input is an `.h5ad` file, open the file in backed mode.

**Returns:**
- `AnnData`: The loaded AnnData object containing gene expression data.

### Function `save_ginseng_state`

Save a Ginseng model state to disk.

**Parameters:**
- **state** (`GinsengModelState`): Model state containing parameters, genes, labels, and metadata.
- **filename** (`str | Path`): Path where the model state will be stored.

### Function `load_ginseng_state`

Load a Ginseng model state.

**Parameters:**
- **filename** (`str | Path`): Path to the HDF5 file containing the saved model state.

**Returns:**
- `GinsengModelState`: Reconstructed model state.

## `nn`

### Function `nn_xavier_uniform`

Initialize weights with Xavier uniform distribution.

**Parameters:**
- **key** (`Array`): PRNG key array for random number generation.
- **shape** (`tuple of int`): Shape of the weight matrix.

**Returns:**
- `Array`: Initialized weight matrix.

### Function `nn_init_linear`

Initialize parameters for a linear layer.

**Parameters:**
- **key** (`Array`): PRNG key array for random initialization.
- **in_dim** (`int`): Input dimension.
- **out_dim** (`int`): Output dimension.

**Returns:**
- `PyTree[Float[Array, "..."]]`: Dictionary with weight matrix `W` and bias vector `b`.

### Function `nn_linear`

Apply a linear transformation.

**Parameters:**
- **params** (`PyTree[Float[Array, "..."]]`): Layer parameters with `W` and `b`.
- **x** (`Array`): Input tensor.

**Returns:**
- `Array`: Transformed output tensor.

### Function `nn_dropout`

Apply dropout to input array.

**Parameters:**
- **key** (`Array`): PRNG key array for dropout mask.
- **x** (`Array`): Input tensor.
- **rate** (`float`): Dropout probability.
- **training** (`bool`): If True, apply dropout. Set to False for determinstic output after training.

**Returns:**
- `Array`: Tensor with dropout applied.

### Function `nn_normalize`

Normalize counts per cell and apply log transform.

**Parameters:**
- **x** (`Array`): Count matrix.
- **target_sum** (`float`): Target total count per cell after normalization.

**Returns:**
- `Array`: Normalized expression matrix.

### Function `nn_annotate_init`

Initialize parameters for the cell type annotation network.

**Parameters:**
- **key** (`Array`): PRNG key array for random initialization.
- **n_genes** (`int`): Number of genes.
- **n_classes** (`int`): Number of classes.
- **hidden_dim** (`int`): Dimension of hidden layers.

**Returns:**
- `PyTree[Float[Array, "..."]]`: Dictionary of initialized model parameters.

### Function `nn_annotate`

Annotate cells using a neural attention-based model.

**Parameters:**
- **params** (`PyTree[Float[Array, "..."]]`): Model parameters from `nn_annotate_init`.
- **key** (`Array`): PRNG key array for random number generation.
- **x** (`Array`): Input gene expression matrix.
- **dropout_rate** (`float`): Dropout probability.
- **normalize** (`bool`): If True, normalize and log-transform counts.
- **target_sum** (`None | float`): Target total count per cell after normalization (defaults to number of genes).
- **return_attn** (`bool`): If True, also return attention weights.
- **training** (`bool`): Whether the model is in training mode.

**Returns:**
- `Float[Array, ...] | tuple[Float[Array, ...], Float[Array, ...]]`: Logits. If `return_attn=True`, gene-level attention weights are also returned.

### Function `nn_annotate_loss`

Cross-entropy loss for cell type annotation model.

**Parameters:**
- **params** (`PyTree[Float[Array, "..."]]`): Model parameters from `nn_annotate_init`.
- **key** (`Array`): PRNG key array for random number generation.
- **x** (`Array`): Input gene expression matrix.
- **y** (`Array`): True cell type labels.
- **dropout_rate** (`float`): Dropout probability.
- **normalize** (`bool`): If True, normalize and log-transform counts.
- **target_sum** (`None | float`): Target total count per cell after normalization (defaults to number of genes).
- **training** (`bool`): Whether the model is in training mode.

**Returns:**
- `Float[Array, ...] | tuple[Float[Array, ...], Float[Array, ...]]`: Logits. If `return_attn=True`, gene-level attention weights are also returned.

### Function `nn_annotate_evaluate`

Cross-entropy loss for cell type annotation model.

**Parameters:**
- **params** (`PyTree[Float[Array, "..."]]`): Model parameters from `nn_annotate_init`.
- **key** (`Array`): PRNG key array for random number generation.
- **x** (`Array`): Input gene expression matrix.
- **y** (`Array`): True cell type labels.
- **dropout_rate** (`float`): Dropout probability.
- **normalize** (`bool`): If True, normalize and log-transform counts.
- **target_sum** (`None | float`): Target total count per cell after normalization (defaults to number of genes).

**Returns:**
- `tuple[float, int, int]`: Loss, total number of labels, and number of correct labels

## `opt`

### Class `AdamState`

State for the Adam optimizer.

**Attributes:**
- **step** (`int`): Current optimization step.
- **m** (`PyTree of Array`): Exponential moving average of gradients.
- **v** (`PyTree of Array`): Exponential moving average of squared gradients.

### Function `opt_init_adam`

Initialize Adam optimizer state.

**Parameters:**
- **params** (`PyTree[Float[Array, ...]]`): Model parameters to be optimized.

**Returns:**
- `AdamState`: Initial optimizer state with zeroed moments.

### Function `opt_adam_update`

Perform one Adam optimization step.

**Parameters:**
- **grads** (`PyTree[Float[Array, ...]]`): Gradients of the loss w.r.t. parameters.
- **params** (`PyTree[Float[Array, ...]]`): Current model parameters.
- **state** (`AdamState`): Current optimizer state.
- **lr** (`float`): Learning rate.
- **betas** (`tuple[float, float]`): Exponential decay rates for first and second moment estimates.
- **eps** (`float`): Numerical stability constant.
- **weight_decay** (`float`): L2 regularization factor.

**Returns:**
- `tuple`: Updated parameters and new optimizer state.

## `train`

### Class `GinsengTrainerSettings`

Training configuration settings for `GinsengTrainer`.

**Attributes:**
- **rate** (`float | None`): Probability of randomly masking input counts.
- **lam_max** (`float | None`): Maximum mean of Poisson distribution for randomly adding counts.
- **lower** (`int | None`): Minimum number of genes to randomly mask out.
- **upper** (`int | None`): Maximum number of genes to randomly mask out.
- **hidden_dim** (`int`): Number of hidden dimensions.
- **dropout_rate** (`float`): Dropout probability applied during training.
- **batch_size** (`int`): Number of samples per training batch.
- **lr** (`float`): Learning rate for Adam optimizer.
- **betas** (`tuple[float, float]`): Exponential decay rates for first and second moment estimates for ADam optimizer.
- **eps** (`float`): Numerical stability constant for Adam optimizer.
- **weight_decay** (`float`): L2 regularization factor for Adam optimizer.
- **normalize** (`bool`): Whether to normalize the input count data.
- **target_sum** (`float`): Target sum used for data normalization.
- **holdout_fraction** (`float`): Fraction of the dataset to hold out for validation.
- **balance_train** (`bool`): Whether to balance training samples across classes.
- **group_level** (`bool`): Whether to apply grouping at the group level.
- **group_mode** (`str`): Strategy for handling group balancing ('fraction' or 'loo').
- **seed** (`int`): Random seed for reproducibility.

### Class `GinsengLogger`

Logger for storing training and validation metrics across epochs.

**Attributes:**
- **epoch** (`list[int]`): List of epoch indices.
- **train_loss** (`list[float]`): Training loss values for each epoch.
- **holdout_loss** (`list[float]`): Holdout loss values for each epoch.
- **holdout_accuracy** (`list[float]`): Holdout accuracy values for each epoch.

#### Method `update`

Update the logger with new training and validation metrics.

**Parameters:**
- **epoch** (`int`): Current epoch index.
- **train_loss** (`float`): Training loss at this epoch.
- **holdout_loss** (`float`): Validation loss at this epoch.
- **holdout_accuracy** (`float`): Validation accuracy at this epoch.

### Class `GinsengModelState`

State of the Ginseng model, including parameters and metadata.

**Attributes:**
- **params** (`PyTree`): Model parameters.
- **genes** (`np.ndarray`): Gene names used during model training.
- **label_keys** (`np.ndarray`): Keys denoting integer identifier for each label.
- **label_values** (`np.ndarray`): Values denoting original string/integer identifiers for each label.
- **normalize** (`bool`): If True, model was trained on normalized data
- **target_sum** (`float`): Target sum used for normalization.
- **training** (`bool`): If True, model weighted should not be frozen.

### Function `GinsengTrainer`

Train a neural network classiifier on a `GinsengDataset`.

**Parameters:**
- **dataset** (`GinsengDataset | str`): GinsengDataset or path to GinsengDataset
- **settings** (`GinsengTrainerSettings`): Training configuration including model, optimization, and augmentation parameters.
- **epochs** (`int`): Number of training epochs to run.
- **silent** (`bool`): If True, suppresses training progress output.

**Returns:**
- `logger : GinsengLogger`: Training log with loss and accuracy metrics across epochs. state : GinsengModelState Final trained model state including parameters and metadata.

## `utils`

### Function `iter_sequential`

Iterate sequentially through an AnnData for prediction.

**Parameters:**
- **adata** (`AnnData`): AnnData object with `obs[label_key]`.
- **label_key** (`str`): Column in AnnData `obs` with integer or string cell type labels.
- **batch_size** (`int`): Number of cells per batch.
- **gene_order** (`None | list | np.ndarray`): Columns ordered by name (e.g as in training). Missing values filled with zero.
- **gene_mask** (`np.ndarray`): Only include columns set to True in the mask. Cannot be set if `gene_order` is provided.

### Function `compute_hvgs`

Select highly variable genes.

**Parameters:**
- **data** (`AnnData`): AnnData object with gene names stored in `var`.
- **n_top_genes** (`int`): Number of top highly variable genes to select.
- **min_mean** (`float`): Lower bound on mean gene expression for selecting highly variable genes.
- **max_mean** (`float`): Upper bound on mean gene expression for selecting highly variable genes.
- **n_bins** (`int`): Select genes across this many gene expression bins.
- **copy** (`bool`): If True, return a copy of the AnnData.

**Returns:**
- `None`: Marks highly variable genes in `var` in-place. Or if copy=True, then returns a copy of the AnnData.
