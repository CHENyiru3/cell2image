"""GenePT vectors, sample-aware HVGs, one-to-one grids, and numerical images."""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import softmax
from sklearn.decomposition import PCA
from umap import UMAP


def load_genept_embeddings(path, expression):
    """Match unique exact symbols to stable expression IDs and L2-normalize.

    Ambiguous symbols and unusable source vectors are omitted. Original GenePT
    coordinates are retained; there is no feature standardization or PCA here.
    """
    source = pd.read_pickle(path)
    source.index = source.index.astype(str)
    genes = expression.var[["gene_symbol"]].copy()
    genes["gene_id"] = expression.var_names.astype(str)
    symbols = genes.gene_symbol
    unique = ~symbols.duplicated(keep=False) & ~symbols.str.contains(";", regex=False)
    source_keys = source.index[~source.index.duplicated(keep=False)]
    genes = genes.loc[unique & symbols.isin(source_keys)].sort_values("gene_id")
    vectors = source.loc[genes.gene_symbol].to_numpy(dtype=np.float32, copy=True)
    usable = np.isfinite(vectors).all(axis=1) & np.any(vectors != 0, axis=1)
    genes = genes.loc[usable].reset_index(drop=True)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[~usable] = 1
    vectors = (vectors / norms).astype(np.float32)
    return genes, np.asfortranarray(vectors[usable])


def select_hvgs(expression, genes, n_genes, exclude_prefixes, exclude_symbols):
    """Select HVGs from matched genes using the existing full-gene normalization."""
    excluded = genes.gene_symbol.str.startswith(tuple(exclude_prefixes))
    excluded |= genes.gene_symbol.isin(exclude_symbols)
    eligible = genes.loc[~excluded].set_index("gene_id", drop=False)
    candidates = expression[:, eligible.gene_id].copy()
    statistics = sc.pp.highly_variable_genes(
        candidates, n_top_genes=n_genes, flavor="seurat", batch_key="sample",
        inplace=False,
    )
    eligible = eligible.join(statistics[[
        "highly_variable", "highly_variable_nbatches", "dispersions_norm",
    ]])
    return eligible.loc[eligible.highly_variable].reset_index(drop=True).sort_values(
        ["highly_variable_nbatches", "dispersions_norm", "gene_id"],
        ascending=[False, False, True], kind="stable",
    ).reset_index(drop=True)


def project_genes(embedding, seed=42, **parameters):
    """Fixed cosine UMAP of the high-dimensional gene vectors."""
    return UMAP(
        n_components=2, metric="cosine", init="spectral", n_jobs=1,
        random_state=seed, transform_seed=seed, low_memory=True, **parameters,
    ).fit_transform(embedding)


def fit_grid_mapping(genes, projection):
    """Exact squared-cost assignment of a fixed projection to distinct pixels.

    The dense solver is intended for the 2,025-gene image subset. It is not a
    high-dimensional Gromov–Wasserstein optimization.
    """
    side = int(np.ceil(np.sqrt(len(genes))))
    center = (projection.max(axis=0) + projection.min(axis=0)) / 2
    scale = np.ptp(projection, axis=0).max()
    points = (projection - center) / scale + 0.5
    grid = np.stack(np.meshgrid(
        np.arange(side), np.arange(side), indexing="ij",
    ), axis=-1).reshape(-1, 2) / (side - 1)
    _, pixels = linear_sum_assignment(cdist(points, grid, "sqeuclidean"))
    mapping = genes.copy()
    mapping["embedding_row"] = np.arange(len(genes))
    mapping["grid_row"] = pixels // side
    mapping["grid_column"] = pixels % side
    mapping["pixel_index"] = pixels
    mapping[["projection_0", "projection_1"]] = points
    occupancy = np.zeros(side * side, dtype=bool)
    occupancy[pixels] = True
    return mapping, occupancy.reshape(side, side)


def fit_gene_colors(embedding, temperature=2.0, seed=42):
    """softmax(PCA3(E)/temperature), with deterministic loading signs."""
    pca = PCA(n_components=3, svd_solver="full", random_state=seed)
    scores = pca.fit_transform(embedding)
    signs = np.sign(pca.components_[np.arange(3), np.argmax(abs(pca.components_), axis=1)])
    signs[signs == 0] = 1
    pca.components_ *= signs[:, None]
    scores *= signs
    weights = softmax(scores / temperature, axis=1).astype(np.float32)
    parameters = {
        "components": pca.components_, "mean": pca.mean_, "scores": scores,
        "explained_variance": pca.explained_variance_,
        "component_scaling": np.ones(3), "temperature": temperature,
    }
    return weights, parameters


def encode_cell_images(expression, weights, pixels, side):
    """Return a float32 batch: expression[cell, gene] × fixed RGB weights."""
    values = expression.toarray() if sparse.issparse(expression) else np.asarray(expression)
    values = np.atleast_2d(values)
    images = np.zeros((len(values), side * side, 3), dtype=np.float32)
    images[:, pixels, :] = values[:, :, None] * weights[None, :, :]
    return images.reshape(-1, side, side, 3)


def decode_expression(images, pixels):
    """Recover retained-gene expression in mapping order by summing channels."""
    return images.reshape(len(images), -1, 3)[:, pixels, :].sum(axis=-1)


def gene_mask_to_pixel_mask(gene_mask, pixels, side):
    """Map selected genes to the same occupied pixels in all three channels."""
    mask = np.zeros(side * side, dtype=bool)
    mask[np.asarray(pixels)[np.asarray(gene_mask, dtype=bool)]] = True
    return np.repeat(mask.reshape(side, side, 1), 3, axis=-1)


def image_batches(expression, mapping, weights, rows=None, batch_size=32, layer="log"):
    """Generate images lazily without materializing every cell in memory."""
    rows = np.arange(expression.n_obs) if rows is None else np.asarray(rows)
    columns = expression.var_names.get_indexer(mapping.gene_id)
    matrix = expression.layers["counts"] if layer == "raw" else expression.X
    side = int(np.ceil(np.sqrt(len(mapping))))
    for start in range(0, len(rows), batch_size):
        indices = rows[start:start + batch_size]
        yield indices, encode_cell_images(
            matrix[indices][:, columns], weights, mapping.pixel_index.to_numpy(), side,
        )


def display_image(values, weights, pixels, side, scale, mode="rgb"):
    """Separate display transform with one shared global scale."""
    brightness = np.clip(np.asarray(values, dtype=np.float32) / scale, 0, 1)
    color = weights / weights.max(axis=1, keepdims=True)
    if mode == "grayscale":
        color = np.ones_like(color)
    elif mode in ("red", "green", "blue"):
        channel = ("red", "green", "blue").index(mode)
        color = np.repeat(color[:, channel:channel + 1], 3, axis=1)
    return encode_cell_images(brightness, color, pixels, side)[0]
