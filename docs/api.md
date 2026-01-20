# API Reference

## Table of Contents

- [data/dataset.py](#datadatasetpy)
    - [Class: GinsengDataset](#class-ginsengdataset)
        - [Method: _load_metadata](#method--load-metadata)
        - [Method: create](#method-create)
        - [Method: make_split](#method-make-split)
        - [Method: stream](#method-stream)
- [data/io.py](#dataiopy)
    - [Function: read_10x_mtx](#function-read-10x-mtx)
    - [Function: read_10x_h5](#function-read-10x-h5)
    - [Function: read_adata](#function-read-adata)
    - [Function: _load_pytree](#function--load-pytree)
    - [Function: save_model](#function-save-model)
    - [Function: _save_pytree](#function--save-pytree)
    - [Function: load_model](#function-load-model)
- [model/nn.py](#modelnnpy)
    - [Class: GinsengClassifier](#class-ginsengclassifier)
        - [Method: _get_key](#method--get-key)
        - [Method: predict](#method-predict)
        - [Method: loss](#method-loss)
        - [Method: evaluate](#method-evaluate)
    - [Function: nn_xavier_uniform](#function-nn-xavier-uniform)
    - [Function: nn_init_linear](#function-nn-init-linear)
    - [Function: nn_linear](#function-nn-linear)
    - [Function: nn_dropout](#function-nn-dropout)
    - [Function: nn_normalize](#function-nn-normalize)
    - [Function: nn_annotate_init](#function-nn-annotate-init)
    - [Function: nn_annotate](#function-nn-annotate)
    - [Function: nn_annotate_loss](#function-nn-annotate-loss)
    - [Function: nn_annotate_evaluate](#function-nn-annotate-evaluate)
- [model/predict.py](#modelpredictpy)
    - [Function: _get_gene_indices](#function--get-gene-indices)
    - [Function: annotate_iter](#function-annotate-iter)
    - [Function: classify](#function-classify)
- [model/state.py](#modelstatepy)
    - [Class: GinsengClassifierState](#class-ginsengclassifierstate)
    - [Function: classifier_from_state](#function-classifier-from-state)
    - [Function: state_from_classifier_trainer](#function-state-from-classifier-trainer)
- [train/augment.py](#trainaugmentpy)
    - [Function: augment_mask](#function-augment-mask)
    - [Function: augment_background](#function-augment-background)
    - [Function: augment_dropgene](#function-augment-dropgene)
    - [Function: augment](#function-augment)
- [train/logger.py](#trainloggerpy)
    - [Class: GinsengLogger](#class-ginsenglogger)
        - [Method: update](#method-update)
        - [Method: report](#method-report)
- [train/opt.py](#trainoptpy)
    - [Class: AdamState](#class-adamstate)
    - [Function: opt_init_adam](#function-opt-init-adam)
    - [Function: opt_adam_update](#function-opt-adam-update)
- [train/trainer.py](#traintrainerpy)
    - [Class: GinsengClassifierTrainerSettings](#class-ginsengclassifiertrainersettings)
    - [Class: GinsengClassifierTrainer](#class-ginsengclassifiertrainer)
        - [Method: _train_step](#method--train-step)
        - [Method: fit](#method-fit)
        - [Method: _run_epoch](#method--run-epoch)
        - [Method: _validate](#method--validate)
- [utils/hvg.py](#utilshvgpy)
    - [Function: select_hvgs](#function-select-hvgs)

---

## <a name="datadatasetpy"></a>File: `data/dataset.py`

### <a name="class-ginsengdataset"></a>Class `GinsengDataset`

An on-disk dataset for training single-cell classifiers.

**Attributes:**

  - **path** (`Path`): Path to the zarr dataset on disk.
  - **root** (`zarr.Group`): The root zarr group object.
  - **n_cells** (`int`): Total number of cells (observations) in the dataset.
  - **n_genes** (`int`): Total number of genes (variables) in the dataset.
  - **label_names** (`list of str`): Human-readable names for the integer labels.
  - **gene_names** (`list of str`): Names of the genes stored in the dataset.
  - **labels** (`np.ndarray`): Integer labels for every cell in the dataset.
  - **groups** (`np.ndarray or None`): Categorical group indices (e.g., batch or donor) if provided.
  - **train_idx** (`np.ndarray or None`): Indices of cells assigned to the training split.
  - **test_idx** (`np.ndarray or None`): Indices of cells assigned to the test split.

#### <a name="method--load-metadata"></a>Method `_load_metadata`

Load metadata and small arrays into memory.

#### <a name="method-create"></a>Method `create`

Create a GinsengDataset from an AnnData object or file path.

**Parameters:**

  - **path** (`str | Path`): Output path where the zarr dataset directory will be created.
  - **adata** (`str | Path | ad.AnnData`): Input data. Can be an AnnData object, a local path to a (.h5ad, .h5, or 10x directory), or a URL to a supported file format.
  - **label_key** (`str`): The column name in `adata.obs` containing the target labels (e.g., cell type).
  - **layer** (`str, optional`): The key in `adata.layers` to use for expression counts. If None, uses `adata.X` (default : None).
  - **genes** (`str | list of str | np.ndarray, optional`): Gene selection/filtering logic. - If a string: Assumes it is a column in `adata.var` containing a boolean mask (e.g., "highly_variable"). - If a list or array: A specific set of gene names to keep. This will also reorder the output to match the provided list. - If None: Keeps all genes (default : None).
  - **group_key** (`str, optional`): The column name in `adata.obs` containing grouping metadata, such as "batch" or "donor" (default : None).
  - **chunk_size** (`int`): Number of rows (cells) per zarr chunk for the expression matrix. Larger chunks improve compression but require more RAM during streaming (default : 4096).
  - **overwrite** (`bool`): Whether to delete the existing directory at `path` if it exists (default : True).

**Returns:**

  - `GinsengDataset`: An initialized instance of the dataset pointing to the new zarr store.

#### <a name="method-make-split"></a>Method `make_split`

Create train/test splits and store indices on disk.

**Parameters:**

  - **fraction** (`float`): The proportion of data (or groups) to include in the test split. Must be in the range [0.0, 1.0) (default : 0.1).
  - **stratify_group** (`bool`): If True, splits by groups (e.g., donors) rather than individual cells. Requires group_key to have been provided during creation (default : False).
  - **seed** (`int`): Random seed for reproducibility (default : 123).

#### <a name="method-stream"></a>Method `stream`

Stream mini-batches of (expression, label) from the zarr store.

**Parameters:**

  - **batch_size** (`int`): Number of cells to yield in each batch.
  - **split** (`Literal["train", "test", "all"]`): Which data split to stream from (default : "train").
  - **balance_labels** (`bool`): If True, downsamples the training split to match the frequency of the least common class (default : False).
  - **shuffle** (`bool`): Whether to shuffle the indices before streaming (default : True).  Yields ------
  - **X** (`np.ndarray`): Expression matrix batch of shape (batch_size, n_genes).
  - **y** (`np.ndarray`): Integer labels batch of shape (batch_size,).

## <a name="dataiopy"></a>File: `data/io.py`

### <a name="function-read-10x-mtx"></a>Function `read_10x_mtx`

Read 10x Genomics mtx format into an AnnData object.

**Parameters:**

  - **path** (`str`): Path to directory containing matrix.mtx, barcodes.tsv, and genes.tsv/features.tsv.
  - **var_names** (`str`): Select 'gene_symbols' or 'gene_ids' as index for var.
  - **make_unique** (`bool`): If True, make var_names unique.

**Returns:**

  - `AnnData`: 

### <a name="function-read-10x-h5"></a>Function `read_10x_h5`

Read 10x Genomics h5 format into an AnnData object.

**Parameters:**

  - **path** (`str`): Path to 10x Genomics .h5 file.
  - **genome** (`str, optional`): Genome to extract (if file contains multiple genomes). If None, will use the first genome available.
  - **var_names** (`str`): Select 'gene_symbols' or 'gene_ids' as index for var.
  - **make_unique** (`bool`): If True, make var_names unique.

**Returns:**

  - `AnnData`: 

### <a name="function-read-adata"></a>Function `read_adata`

Read an AnnData object from various supported file formats.

**Parameters:**

  - **path** (`str | Path`): Path to the input count data stored in 10x .h5 format, AnnData .h5ad format, or in a 10x matrix market format folder.
  - **backed** (`bool`): If True and the input is an `.h5ad` file, open the file in backed mode.

**Returns:**

  - `AnnData`: The loaded AnnData object containing gene expression data.

### <a name="function--load-pytree"></a>Function `_load_pytree`

Recursively load a PyTree from an HDF5 group.

### <a name="function-save-model"></a>Function `save_model`

Save a Ginseng model state to a single HDF5 file.

**Parameters:**

  - **state** (`GinsengClassifierState`): Complete model state to save.
  - **path** (`str | Path`): Path to the output HDF5 file (will add .h5 if not present).

**Examples:**

```python
>>> # After training
>>> state = GinsengClassifierState(
...     params=model.params,
...     genes=dataset.gene_names,
...     label_keys=dataset.label_names,
...     label_values=np.arange(len(dataset.label_names)),
...     n_genes=dataset.n_genes,
...     n_classes=len(dataset.label_names),
...     hidden_dim=256,
...     normalize=True,
...     target_sum=1e4,
...     dropout_rate=0.5,
...     training=False,
... )
>>> save_model(state, "./models/my_classifier.h5")
```

### <a name="function--save-pytree"></a>Function `_save_pytree`

Recursively save a PyTree to an HDF5 group.

### <a name="function-load-model"></a>Function `load_model`

Load a Ginseng model state from an HDF5 file.

**Parameters:**

  - **path** (`str | Path`): Path to the HDF5 file.

**Returns:**

  - `GinsengClassifierState`: Complete model state ready for inference or continued training.

**Examples:**

```python
>>> state = load_model("./models/my_classifier.h5")
>>>
>>> # Recreate model
>>> model = GinsengClassifier(
...     n_genes=state.n_genes,
...     n_classes=state.n_classes,
...     hidden_dim=state.hidden_dim,
...     dropout_rate=state.dropout_rate,
...     normalize=state.normalize,
...     target_sum=state.target_sum,
... )
>>> model.params = state.params
>>>
>>> # Prepare new data with correct gene order
>>> new_data_ordered = new_data[state.genes]
>>> predictions = model.predict(new_data_ordered.values, training=False)
```

## <a name="modelnnpy"></a>File: `model/nn.py`

### <a name="class-ginsengclassifier"></a>Class `GinsengClassifier`

Wrapper class for cell type annotation with automatic key management.

**Attributes:**

  - **n_genes** (`int`): Number of genes in input data.
  - **n_classes** (`int`): Number of cell type classes.
  - **hidden_dim** (`int`): Hidden dimension for attention mechanism (default: 256).
  - **dropout_rate** (`float`): Dropout rate during training (default: 0.5).
  - **normalize** (`bool`): Whether to normalize input data (default: True).
  - **target_sum** (`None | float`): Target sum for normalization (default: 1e4).
  - **seed** (`int`): Random seed for reproducibility (default: 42).

**Examples:**

```python
>>> model = GinsengClassifier(n_genes=2000, n_classes=10)
>>> # During training (JAX-compatible)
>>> loss = model.loss(model.params, model._get_key(), x_batch, y_batch)
>>> # For prediction (Standard usage)
>>> logits = model.predict(x_test, training=False)
```

#### <a name="method--get-key"></a>Method `_get_key`

Get a new PRNG key and update internal state.

**Returns:**

  - `Array`: A new JAX PRNG key.

#### <a name="method-predict"></a>Method `predict`

Generate predictions for input data.

**Parameters:**

  - **x** (`Array`): Input gene expression matrix.
  - **params** (`PyTree, optional`): Model parameters. If None, uses internal self.params.
  - **key** (`Array, optional`): PRNG key. If None, uses a new key from self._get_key().
  - **training** (`bool`): Whether to use training mode (with dropout).
  - **return_attn** (`bool`): Whether to return attention weights.

**Returns:**

  - `Array or tuple`: Logits, or (logits, attention) if return_attn=True.

#### <a name="method-loss"></a>Method `loss`

Compute cross-entropy loss for a batch.

**Parameters:**

  - **params** (`PyTree`): Model parameters to differentiate against.
  - **key** (`Array`): PRNG key for dropout randomness.
  - **x** (`Array`): Input gene expression matrix.
  - **y** (`Array`): True class labels.

**Returns:**

  - `Array`: Scalar loss value.

#### <a name="method-evaluate"></a>Method `evaluate`

Evaluate model on a batch and return loss and accuracy.

**Parameters:**

  - **x** (`Array`): Input gene expression matrix.
  - **y** (`Array`): True class labels.

**Returns:**

  - `tuple[float, float]`: (loss, accuracy) on the batch.

### <a name="function-nn-xavier-uniform"></a>Function `nn_xavier_uniform`

Initialize weights with Xavier uniform distribution.

**Parameters:**

  - **key** (`Array`): PRNG key array for random number generation.
  - **shape** (`tuple of int`): Shape of the weight matrix.

**Returns:**

  - `Array`: Initialized weight matrix.

### <a name="function-nn-init-linear"></a>Function `nn_init_linear`

Initialize parameters for a linear layer.

**Parameters:**

  - **key** (`Array`): PRNG key array for random initialization.
  - **in_dim** (`int`): Input dimension.
  - **out_dim** (`int`): Output dimension.

**Returns:**

  - `PyTree[Float[Array, "..."]]`: Dictionary with weight matrix `W` and bias vector `b`.

### <a name="function-nn-linear"></a>Function `nn_linear`

Apply a linear transformation.

**Parameters:**

  - **params** (`PyTree[Float[Array, "..."]]`): Layer parameters with `W` and `b`.
  - **x** (`Array`): Input tensor.

**Returns:**

  - `Array`: Transformed output tensor.

### <a name="function-nn-dropout"></a>Function `nn_dropout`

Apply dropout to input array with optimized implementation.

**Parameters:**

  - **key** (`Array`): PRNG key array for dropout mask.
  - **x** (`Array`): Input tensor.
  - **rate** (`float`): Dropout probability.
  - **training** (`bool`): If True, apply dropout. Set to False for deterministic output after training.

**Returns:**

  - `Array`: Tensor with dropout applied.

### <a name="function-nn-normalize"></a>Function `nn_normalize`

Normalize counts per cell and apply log transform.

**Parameters:**

  - **x** (`Array`): Count matrix.
  - **target_sum** (`float`): Target total count per cell after normalization.

**Returns:**

  - `Array`: Normalized expression matrix.

### <a name="function-nn-annotate-init"></a>Function `nn_annotate_init`

Initialize parameters for the cell type annotation network.

**Parameters:**

  - **key** (`Array`): PRNG key array for random initialization.
  - **n_genes** (`int`): Number of genes.
  - **n_classes** (`int`): Number of classes.
  - **hidden_dim** (`int`): Dimension of hidden layers for attention mechanism.

**Returns:**

  - `PyTree[Float[Array, "..."]]`: Dictionary of initialized model parameters.

### <a name="function-nn-annotate"></a>Function `nn_annotate`

Annotate cells using instance-based attention neural network.

**Parameters:**

  - **params** (`PyTree[Float[Array, "..."]]`): Model parameters from `nn_annotate_init`.
  - **key** (`Array`): PRNG key array for random number generation.
  - **x** (`Array`): Input gene expression matrix (batch_size, n_genes).
  - **dropout_rate** (`float`): Dropout probability (default: 0.5).
  - **normalize** (`bool`): If True, normalize and log-transform counts (default: True).
  - **target_sum** (`None | float`): Target total count per cell after normalization (default: 1e4).
  - **return_attn** (`bool`): If True, also return attention weights (default: False).
  - **training** (`bool`): Whether the model is in training mode (default: True).

**Returns:**

  - `Float[Array, ...] | tuple[Float[Array, ...], Float[Array, ...]]`: Logits for each class. If `return_attn=True`, gene-level attention weights are also returned as second element.

### <a name="function-nn-annotate-loss"></a>Function `nn_annotate_loss`

Cross-entropy loss for cell type annotation model.

**Parameters:**

  - **params** (`PyTree[Float[Array, "..."]]`): Model parameters from `nn_annotate_init`.
  - **key** (`Array`): PRNG key array for random number generation.
  - **x** (`Array`): Input gene expression matrix (batch_size, n_genes).
  - **y** (`Array`): True cell type labels as integer class indices (batch_size,).
  - **dropout_rate** (`float`): Dropout probability (default: 0.5).
  - **normalize** (`bool`): If True, normalize and log-transform counts (default: True).
  - **target_sum** (`None | float`): Target total count per cell after normalization. Defaults to 1e4 (standard for scRNA-seq).

**Returns:**

  - `Float[Array, ""]`: Scalar cross-entropy loss.

### <a name="function-nn-annotate-evaluate"></a>Function `nn_annotate_evaluate`

Evaluate model performance on a batch.

**Parameters:**

  - **params** (`PyTree[Float[Array, "..."]]`): Model parameters from `nn_annotate_init`.
  - **key** (`Array`): PRNG key array for random number generation.
  - **x** (`Array`): Input gene expression matrix (batch_size, n_genes).
  - **y** (`Array`): True cell type labels as integer class indices (batch_size,).
  - **dropout_rate** (`float`): Dropout probability (default: 0.5).
  - **normalize** (`bool`): If True, normalize and log-transform counts (default: True).
  - **target_sum** (`None | float`): Target total count per cell after normalization. Defaults to 1e4 (standard for scRNA-seq).

**Returns:**

  - `tuple[Float[Array, ""], Int[Array, ""], Int[Array, ""]]`: Tuple of (loss, total_samples, correct_predictions).

## <a name="modelpredictpy"></a>File: `model/predict.py`

### <a name="function--get-gene-indices"></a>Function `_get_gene_indices`

Map model gene names to indices within the AnnData object.

**Parameters:**

  - **adata** (`AnnData`): The single-cell data object.
  - **gene_names** (`np.ndarray`): Array of gene names expected by the model in a specific order.
  - **gene_key** (`str | None`): Column name in `adata.var` containing gene names. If None, uses `adata.var_names`.

**Returns:**

  - `available_idx`: np.ndarray The integer indices of genes found in `adata` that match the model's expected genes. out_positions : np.ndarray The corresponding positions in the model's input vector where these genes belong.

### <a name="function-annotate-iter"></a>Function `annotate_iter`

Batch iterator that handles gene reordering and zero-padding for missing genes.

**Parameters:**

  - **adata** (`AnnData`): The single-cell data object.
  - **gene_names** (`np.ndarray`): Ordered gene names from the trained model.
  - **gene_key** (`str | None`): Column in `adata.var` to use for gene matching.
  - **layer** (`str | None`): Key in `adata.layers` to use for counts. If None, uses `adata.X`.
  - **batch_size** (`int`): Number of cells to process per batch.  Yields ------
  - **batch_tensor** (`jax.numpy.ndarray`): A JAX-compatible array of shape (batch_size, n_model_genes).
  - **start** (`int`): Starting observation index.
  - **end** (`int`): Ending observation index.

### <a name="function-classify"></a>Function `classify`

Annotate single-cell sequencing data using a trained ginseng classifier.

**Parameters:**

  - **model_state** (`GinsengClassifierState | str | Path`): A loaded GinsengClassifierState or a path to a saved state file.
  - **adata** (`AnnData | str | Path`): AnnData object or path to count data (.h5ad, .h5, or Matrix Market).
  - **gene_key** (`str | None`): Column in `.var` containing gene names. If None, uses index.
  - **layer** (`str | None`): Key in `adata.layers` to use for counts. If None, uses `adata.X`.
  - **backed** (`bool`): If True and `adata` is a path, reads data in backed mode.
  - **normalize** (`bool | None`): Override model normalization setting.
  - **target_sum** (`float | None`): Override model target sum.
  - **randomness** (`bool`): If True, enables dropout during inference.
  - **batch_size** (`int`): Number of cells to process in each forward pass.
  - **copy** (`bool`): If True, returns a modified copy of the AnnData object.
  - **store_probs** (`bool`): If True, stores the full probability matrix in `adata.obsm`.
  - **return_table** (`bool`): If True, returns a pandas DataFrame instead of modifying AnnData.
  - **seed** (`int`): Random seed.
  - **silent** (`bool`): If True, suppresses progress bar and warnings.

**Returns:**

  - `AnnData | pd.DataFrame | None`: Returns a DataFrame if `return_table` is True, a copy of AnnData if `copy` is True, otherwise modifies in-place and returns None.

## <a name="modelstatepy"></a>File: `model/state.py`

### <a name="class-ginsengclassifierstate"></a>Class `GinsengClassifierState`

Complete state of a trained Ginseng model.

All information needed to save, load, and use a trained model.

**Attributes:**

  - **params** (`PyTree`): Model parameters (weights and biases).
  - **genes** (`np.ndarray`): Gene names in the exact order expected by the model.
  - **label_keys** (`np.ndarray`): Label names (e.g., ['T-cell', 'B-cell', 'Macrophage']).
  - **label_values** (`np.ndarray`): Integer values corresponding to each label (e.g., [0, 1, 2]).
  - **n_genes** (`int`): Number of genes.
  - **n_classes** (`int`): Number of classes.
  - **hidden_dim** (`int`): Hidden dimension used in attention mechanism.
  - **normalize** (`bool`): Whether input data should be normalized.
  - **target_sum** (`float`): Target sum for normalization.
  - **dropout_rate** (`float`): Dropout rate used during training.
  - **training** (`bool`): Whether weights should be frozen (False after training).

### <a name="function-classifier-from-state"></a>Function `classifier_from_state`

Create a GinsengClassifier from a loaded state.

**Parameters:**

  - **state** (`GinsengClassifierState`): Loaded model state.

**Returns:**

  - `GinsengClassifier`: Model ready for inference.

### <a name="function-state-from-classifier-trainer"></a>Function `state_from_classifier_trainer`

Create a GinsengClassifierState from a trainer after .fit().

**Parameters:**

  - **trainer** (`GinsengClassifierTrainer`): Trainer instance with a trained model.

**Returns:**

  - `GinsengClassifierState`: Complete model state.

**Examples:**

```python
>>> trainer = GinsengClassifierTrainer(dataset, settings)
>>> trainer.fit(epochs=50)
>>> state = state_from_trainer(trainer)
>>> save_model(state, "./models/my_classifier.h5")
```

## <a name="trainaugmentpy"></a>File: `train/augment.py`

### <a name="function-augment-mask"></a>Function `augment_mask`

Randomly mask out counts across cells.

**Parameters:**

  - **key** (`Array`): PRNG key array for dropout mask.
  - **x** (`Array`): Input tensor.
  - **rate** (`float`): Dropout probability.

**Returns:**

  - `Array`: Tensor with dropout applied.

### <a name="function-augment-background"></a>Function `augment_background`

Randomly add Poisson-distributed background noise to counts.

**Parameters:**

  - **key** (`Array`): PRNG key array for noise generation.
  - **x** (`Array`): Non-normalized count matrix.
  - **lam_max** (`float`): Maximum mean of Poisson distribution for sampling added noise.

**Returns:**

  - `Array`: Counts with added Poisson noise.

### <a name="function-augment-dropgene"></a>Function `augment_dropgene`

Randomly zero out entire genes.

**Parameters:**

  - **key** (`Array`): PRNG key array for mask generation.
  - **x** (`Array`): Count matrix.
  - **lower** (`int`): Minimum number of genes to mask out.
  - **upper** (`int`): Maximum (exclusive) number of genes to mask out.

**Returns:**

  - `Array`: Filtered counts.

### <a name="function-augment"></a>Function `augment`

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

## <a name="trainloggerpy"></a>File: `train/logger.py`

### <a name="class-ginsenglogger"></a>Class `GinsengLogger`

Logger for storing training and validation metrics across epochs.

**Attributes:**

  - **epoch** (`list[int]`): List of epoch indices.
  - **train_loss** (`list[float]`): Training loss values for each epoch.
  - **holdout_loss** (`list[float]`): Holdout loss values for each epoch.
  - **holdout_accuracy** (`list[float]`): Holdout accuracy values for each epoch.

#### <a name="method-update"></a>Method `update`

Update the logger with new training and validation metrics.

**Parameters:**

  - **epoch** (`int`): Current epoch index.
  - **train_loss** (`float`): Training loss at this epoch.
  - **holdout_loss** (`float`): Validation loss at this epoch.
  - **holdout_accuracy** (`float`): Validation accuracy at this epoch.

#### <a name="method-report"></a>Method `report`

Print most recent result to standard output.

**Parameters:**

  - **silent** (`bool`): If True, suppresses report output.
  - **flush** (`bool`): If True, write report to output immediately.

**Returns:**

  - `None`: Output is printed to standard output.

## <a name="trainoptpy"></a>File: `train/opt.py`

### <a name="class-adamstate"></a>Class `AdamState`

State for the Adam optimizer.

**Attributes:**

  - **step** (`int`): Current optimization step.
  - **m** (`PyTree of Array`): Exponential moving average of gradients.
  - **v** (`PyTree of Array`): Exponential moving average of squared gradients.

### <a name="function-opt-init-adam"></a>Function `opt_init_adam`

Initialize Adam optimizer state.

**Parameters:**

  - **params** (`PyTree[Float[Array, ...]]`): Model parameters to be optimized.

**Returns:**

  - `AdamState`: Initial optimizer state with zeroed moments.

### <a name="function-opt-adam-update"></a>Function `opt_adam_update`

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

## <a name="traintrainerpy"></a>File: `train/trainer.py`

### <a name="class-ginsengclassifiertrainersettings"></a>Class `GinsengClassifierTrainerSettings`

Training configuration settings for `GinsengClassifierTrainer`.

### <a name="class-ginsengclassifiertrainer"></a>Class `GinsengClassifierTrainer`

Trainer class for orchestrating the training of a GinsengClassifier.

#### <a name="method--train-step"></a>Method `_train_step`

Internal training step to update parameters.

#### <a name="method-fit"></a>Method `fit`

Execute the training loop.

**Parameters:**

  - **epochs** (`int`): Number of training epochs to run (default : 10).
  - **silent** (`bool`): If True, suppresses training progress output (default : False).

**Returns:**

  - `GinsengClassifier`: The trained classifier with updated parameters.

#### <a name="method--run-epoch"></a>Method `_run_epoch`

Run a single training epoch.

#### <a name="method--validate"></a>Method `_validate`

Run validation on the holdout split.

## <a name="utilshvgpy"></a>File: `utils/hvg.py`

### <a name="function-select-hvgs"></a>Function `select_hvgs`

Select highly variable genes from raw or normalized counts.

**Parameters:**

  - **adata** (`AnnData`): AnnData object with gene names stored in `var`.
  - **n_top_genes** (`int`): Number of top highly variable genes to select.
  - **layer** (`str, optional`): Key in `adata.layers` to use. If None, uses `adata.X`.
  - **target_sum** (`float, optional`): If provided, scales each cell to this sum and applies log1p transformation (default: 1e4).
  - **min_mean** (`float`): Lower quantile bound on mean gene expression.
  - **max_mean** (`float`): Upper quantile bound on mean gene expression.
  - **n_bins** (`int`): Select genes across this many gene expression bins.
  - **chunk_size** (`int`): Number of cells to process in memory at once.
  - **silent** (`bool`): If True, suppresses progress bar.
  - **copy** (`bool`): If True, returns a copy of the AnnData.

**Returns:**

  - `Optional[AnnData]`: Marks highly variable genes in `var['ginseng_genes']`.
