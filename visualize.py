"""Core gene UMAPs, image panels, and self-contained HTML viewers."""

import base64
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from core import display_image, encode_cell_images


def save_figure(figure, path):
    for extension in ("png", "pdf"):
        figure.savefig(path.with_suffix(f".{extension}"), dpi=200, bbox_inches="tight")
    plt.close(figure)


def show_pixels(axis, image, occupancy, title=""):
    image = image.copy()
    image[~occupancy] = 0.85
    axis.imshow(image, interpolation="nearest", vmin=0, vmax=1, cmap="gray")
    axis.set(xticks=[], yticks=[], title=title)


def cell_label(cell):
    return (
        f"{cell.name}\n{cell['sample']} · Leiden {cell['leiden']}\n"
        f"counts {cell.total_counts:,.0f} · genes {cell.n_genes_by_counts:,}"
    )


def embedding_plot(source, destination, seed):
    projection = pd.read_parquet(source / "gene_projection.parquet")
    weights = pd.read_parquet(source / "gene_rgb_weights.parquet")[["red", "green", "blue"]].to_numpy()
    rgb = weights / weights.max(axis=1, keepdims=True)
    order = np.random.default_rng(seed).permutation(len(projection))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor("#20262b")
    xy = projection[["umap_1", "umap_2"]].to_numpy()
    ax.scatter(*xy[order].T, c=rgb[order], s=4, linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="Gene UMAP 1", ylabel="Gene UMAP 2",
           title=f"GenePT · {len(projection):,} genes · all points shown")
    fig.text(0.5, 0.015, "Embedding-derived colors; no expression modulation", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, destination)
    return projection, weights


def write_html(template_name, destination, data):
    template = Path(__file__).with_name(template_name).read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    destination.write_text(template.replace("__DATA__", payload))


def render_visualizations(out, config):
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    full, full_weights = embedding_plot(out / "full_embedding", figures / "01_full_gene_umap", config["seed"])
    embedding_plot(out, figures / "02_hvg_gene_umap", config["seed"])
    mapping = pd.read_parquet(out / "gene_grid_mapping.parquet")
    weights = pd.read_parquet(out / "gene_rgb_weights.parquet")[["red", "green", "blue"]].to_numpy()
    occupancy = np.load(out / "occupancy_mask.npy")
    pixels = mapping.pixel_index.to_numpy()
    random_pixels = np.load(out / "random_pixel_indices.npy")
    side, scale = len(occupancy), config["display_scale"]
    values = np.load(out / "images/reference_log.npy", mmap_mode="r")
    raw = np.load(out / "images/reference_raw.npy", mmap_mode="r")
    cells = pd.read_parquet(out / "images/cells.parquet")

    rgb = weights / weights.max(axis=1, keepdims=True)
    atlas = encode_cell_images(np.ones(len(mapping)), rgb, pixels, side)[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].set_facecolor("#20262b")
    axes[0].scatter(mapping.projection_1, mapping.projection_0, c=rgb, s=5, linewidths=0, rasterized=True)
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal")
    axes[0].set(xlabel="Projected column coordinate", ylabel="Projected row coordinate", title="Fixed UMAP")
    show_pixels(axes[1], atlas, occupancy, f"One gene per pixel · {side} × {side}")
    fig.tight_layout()
    save_figure(fig, figures / "03_projection_and_grid")

    fig, ax = plt.subplots(figsize=(6, 6))
    show_pixels(ax, atlas, occupancy, "GenePT color atlas · equal expression intensity")
    fig.tight_layout()
    save_figure(fig, figures / "04_gene_color_atlas")

    fig, axes = plt.subplots(4, 6, figsize=(15, 14))
    for i, ax in enumerate(axes.flat):
        if i < len(cells):
            show_pixels(ax, display_image(values[i], weights, pixels, side, scale), occupancy, cell_label(cells.iloc[i]))
        else:
            ax.axis("off")
    fig.suptitle(f"Fixed sample/Leiden-stratified examples · shared L={scale:.3f}", fontsize=14)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.02, top=0.92, hspace=0.50, wspace=0.08)
    save_figure(fig, figures / "05_cell_gallery")

    count = min(4, len(cells))
    fig, axes = plt.subplots(count, 3, figsize=(10, 3 * count), squeeze=False)
    options = [(pixels, "rgb", "OT · RGB"), (pixels, "grayscale", "OT · grayscale"),
               (random_pixels, "rgb", "Random · same RGB")]
    for i in range(count):
        for ax, (placement, mode, title) in zip(axes[i], options):
            show_pixels(ax, display_image(values[i], weights, placement, side, scale, mode), occupancy, title if i == 0 else "")
        axes[i, 0].set_ylabel(cell_label(cells.iloc[i]), fontsize=8)
    fig.suptitle(f"Same cells, genes, colors and scale L={scale:.3f}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, figures / "06_matched_cell_views")

    images = np.load(out / "images/RGB_log.npy", mmap_mode="r")
    fig, axes = plt.subplots(2, 5, figsize=(14, 6), squeeze=False)
    for i in range(2):
        reference = np.zeros(side * side, dtype=np.float32)
        reference[pixels] = values[i]
        panels = [images[i, ..., j] for j in range(3)] + [images[i].sum(-1), reference.reshape(side, side)]
        for ax, panel, title in zip(axes[i], panels, ["R", "G", "B", "R + G + B", "Reference log expression"]):
            handle = ax.imshow(np.ma.masked_where(~occupancy, panel), cmap="magma", vmin=0, vmax=scale, interpolation="nearest")
            ax.set(title=title, xticks=[], yticks=[])
        axes[i, 0].set_ylabel(cell_label(cells.iloc[i]), fontsize=8)
    fig.subplots_adjust(right=0.88, wspace=0.08, hspace=0.25)
    fig.colorbar(handle, cax=fig.add_axes([0.90, 0.2, 0.015, 0.6]), label="Numerical log-expression intensity")
    save_figure(fig, figures / "07_rgb_channels")

    pngs = out / "images/png"
    pngs.mkdir(exist_ok=True)
    count = min(config["viewer_cells"], len(cells))
    for i in range(count):
        image = display_image(values[i], weights, pixels, side, scale)
        image[~occupancy] = 0.85
        plt.imsave(pngs / f"{i:03d}_{cells.index[i].replace(':', '_')}.png", image)

    def packed(array):
        return base64.b64encode(np.asarray(array, dtype="<f4").tobytes()).decode("ascii")

    metadata = cells.iloc[:count].reset_index(names="cell_index")
    metadata["leiden"] = metadata.leiden.astype(str)
    write_html("image_viewer.html", out / "viewer.html", {
        "n": side, "ids": mapping.gene_id.tolist(), "symbols": mapping.gene_symbol.tolist(),
        "ot": pixels.tolist(), "random": random_pixels.tolist(), "weights": packed(weights),
        "log": packed(values[:count]), "raw": packed(raw[:count]),
        "cells": json.loads(metadata.to_json(orient="records")), "L": scale,
    })
    write_html("umap_viewer.html", out / "gene_umap.html", {
        "ids": full.gene_id.tolist(), "symbols": full.gene_symbol.tolist(),
        "xy": full[["umap_1", "umap_2"]].to_numpy().tolist(),
        "weights": full_weights.tolist(),
    })
