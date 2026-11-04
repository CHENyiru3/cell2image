"""Run the compact GenePT workflow; inputs are existing processed expression data."""

import os

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[variable] = "4"

import argparse
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import yaml
from core import (
    fit_gene_colors, fit_grid_mapping, image_batches, load_genept_embeddings,
    project_genes, select_hvgs,
)
from visualize import render_visualizations


def save_embedding(out, genes, embedding, projection, weights, pca):
    out.mkdir(parents=True, exist_ok=True)
    genes.to_parquet(out / "genes.parquet", index=False)
    np.save(out / "gene_embedding.npy", embedding)
    coordinates = genes[["gene_id", "gene_symbol"]].copy()
    coordinates[["umap_1", "umap_2"]] = projection
    coordinates.to_parquet(out / "gene_projection.parquet", index=False)
    colors = genes[["gene_id", "gene_symbol"]].copy()
    colors[["red", "green", "blue"]] = weights
    colors.to_parquet(out / "gene_rgb_weights.parquet", index=False)
    np.savez(out / "color_pca.npz", **pca)


def export_images(expression, mapping, weights, cells_path, out, config):
    ids = pd.read_parquet(cells_path).index[:config["export_cells"]]
    cells = expression.obs.loc[ids].copy()
    rows = expression.obs_names.get_indexer(cells.index)
    cells["expression_row"] = rows
    cells["tensor_row"] = np.arange(len(cells))
    destination = out / "images"
    destination.mkdir(exist_ok=True)
    cells.to_parquet(destination / "cells.parquet")
    columns = expression.var_names.get_indexer(mapping.gene_id)
    side = int(np.ceil(np.sqrt(len(mapping))))
    for layer in ("raw", "log"):
        matrix = expression.layers["counts"] if layer == "raw" else expression.X
        values = matrix[rows][:, columns].toarray().astype(np.float32)
        np.save(destination / f"reference_{layer}.npy", values)
        images = np.lib.format.open_memmap(
            destination / f"RGB_{layer}.npy", mode="w+", dtype=np.float32,
            shape=(len(rows), side, side, 3),
        )
        offset = 0
        for _, batch in image_batches(
            expression, mapping, weights, rows, config["batch_size"], layer,
        ):
            images[offset:offset + len(batch)] = batch
            offset += len(batch)
        images.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--visualize-only", action="store_true", help="Redraw saved results without refitting.")
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    config = yaml.safe_load(config_path.read_text())
    def path(key):
        return (config_path.parent / config[key]).resolve()
    out = path("output")
    out.mkdir(parents=True, exist_ok=True)
    if not arguments.visualize_only:
        print("Loading processed expression and GenePT vectors", flush=True)
        expression = ad.read_h5ad(path("expression"))
        genes, embedding = load_genept_embeddings(path("genept"), expression)
        selected = select_hvgs(
            expression, genes, config["n_genes"],
            config["exclude_prefixes"], config["exclude_symbols"],
        )
        indices = pd.Index(genes.gene_id).get_indexer(selected.gene_id)
        image_embedding = np.asfortranarray(embedding[indices])
        print(f"Image atlas: {len(selected):,} HVGs; full UMAP: {len(genes):,} matched genes", flush=True)
        projection = project_genes(image_embedding, config["seed"], **config["umap"])
        mapping, occupancy = fit_grid_mapping(selected, projection)
        weights, pca = fit_gene_colors(image_embedding, config["temperature"], config["seed"])
        save_embedding(out, selected, image_embedding, projection, weights, pca)
        mapping.to_parquet(out / "gene_grid_mapping.parquet", index=False)
        np.save(out / "occupancy_mask.npy", occupancy)
        np.save(out / "random_pixel_indices.npy", np.random.default_rng(config["seed"]).permutation(mapping.pixel_index))
        export_images(expression, mapping, weights, path("example_cells"), out, config)
        del expression
        print("Projecting the full matched gene embedding", flush=True)
        full_projection = project_genes(embedding, config["seed"], **config["umap"])
        full_weights, full_pca = fit_gene_colors(embedding, config["temperature"], config["seed"])
        save_embedding(out / "full_embedding", genes, embedding, full_projection, full_weights, full_pca)
    print("Rendering figures and offline viewers", flush=True)
    render_visualizations(out, config)
    print(f"Finished: {out}", flush=True)


if __name__ == "__main__":
    main()
