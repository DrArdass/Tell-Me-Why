"""Matplotlib helpers for latent-space and feature inspection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from sklearn.decomposition import PCA
from torch import Tensor


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _flatten_labels(labels: Any | None) -> np.ndarray | None:
    if labels is None:
        return None
    label_array = _to_numpy(labels)
    if label_array.ndim > 1:
        label_array = label_array.reshape(label_array.shape[0], -1)[:, 0]
    return label_array.reshape(-1)


def _project_latents(latents: np.ndarray, projection: str) -> np.ndarray:
    if latents.ndim != 2:
        raise ValueError("Latents must be a 2D array shaped `(n_samples, latent_dim)`.")
    if latents.shape[1] == 1:
        return np.column_stack([latents[:, 0], np.zeros(latents.shape[0])])
    if projection == "first2":
        return latents[:, :2]
    if projection == "pca":
        return PCA(n_components=2).fit_transform(latents)
    raise ValueError("`projection` must be either 'first2' or 'pca'.")


def _sample_points(
    points: np.ndarray,
    labels: np.ndarray | None,
    max_points: int | None,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if max_points is None or points.shape[0] <= max_points:
        return points, labels
    rng = np.random.default_rng(random_state)
    indexes = rng.choice(points.shape[0], size=max_points, replace=False)
    return points[indexes], None if labels is None else labels[indexes]


def plot_latent_gaussian(
    latents: Any,
    labels: Any | None = None,
    *,
    ax: Axes | None = None,
    projection: str = "first2",
    max_points: int | None = 5000,
    random_state: int = 42,
    show_prior: bool = True,
    prior_sigmas: Sequence[float] = (1.0, 2.0, 3.0),
    alpha: float = 0.75,
    cmap: str = "tab10",
) -> tuple[Figure, Axes]:
    """Scatter latent vectors and overlay standard Gaussian prior contours."""

    latent_array = _to_numpy(latents).reshape(len(latents), -1)
    label_array = _flatten_labels(labels)
    points = _project_latents(latent_array, projection=projection)
    points, label_array = _sample_points(points, label_array, max_points, random_state)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    fig = ax.figure

    if label_array is None:
        ax.scatter(points[:, 0], points[:, 1], s=14, alpha=alpha)
    else:
        scatter = ax.scatter(
            points[:, 0],
            points[:, 1],
            c=label_array,
            s=14,
            alpha=alpha,
            cmap=cmap,
        )
        fig.colorbar(scatter, ax=ax, label="label")

    if show_prior:
        for sigma in prior_sigmas:
            ax.add_patch(
                Circle(
                    (0.0, 0.0),
                    radius=sigma,
                    fill=False,
                    linestyle="--",
                    linewidth=1,
                    color="black",
                    alpha=0.35,
                )
            )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.2)
    ax.axvline(0.0, color="black", linewidth=0.8, alpha=0.2)
    ax.set_xlabel("z[0]" if projection == "first2" else "PCA 1")
    ax.set_ylabel("z[1]" if projection == "first2" else "PCA 2")
    ax.set_title("Gaussian latent space")
    ax.set_aspect("equal", adjustable="datalim")
    return fig, ax


def _feature_images(features: Any, channel: int | None) -> np.ndarray:
    feature_array = _to_numpy(features)
    if feature_array.ndim == 4:
        selected_channel = 0 if channel is None else channel
        return feature_array[:, selected_channel, :, :]
    if feature_array.ndim == 3:
        return feature_array
    if feature_array.ndim == 2:
        return feature_array[:, np.newaxis, :]
    raise ValueError(
        "Features must be shaped `(n, c, h, w)`, `(n, h, w)`, or `(n, d)`."
    )


def plot_feature_grid(
    features: Any,
    *,
    n: int = 16,
    columns: int = 4,
    channel: int | None = 0,
    cmap: str = "viridis",
    title: str = "Feature maps",
) -> tuple[Figure, np.ndarray]:
    """Display feature maps or feature vectors in a compact grid."""

    images = _feature_images(features, channel=channel)
    count = min(n, images.shape[0])
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 2.5))
    axes_array = np.asarray(axes).reshape(-1)

    for index, axis in enumerate(axes_array):
        axis.axis("off")
        if index >= count:
            continue
        axis.imshow(images[index], aspect="auto", cmap=cmap)
        axis.set_title(f"feature {index}")

    fig.suptitle(title)
    fig.tight_layout()
    return fig, axes_array


def plot_feature_distributions(
    features: Any,
    *,
    feature_names: Sequence[str] | None = None,
    max_features: int = 12,
    bins: int = 40,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot marginal distributions for vector-like features."""

    feature_array = _to_numpy(features).reshape(len(features), -1)
    count = min(max_features, feature_array.shape[1])

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    fig = ax.figure

    for index in range(count):
        name = (
            feature_names[index]
            if feature_names is not None and index < len(feature_names)
            else f"feature {index}"
        )
        ax.hist(
            feature_array[:, index],
            bins=bins,
            histtype="step",
            linewidth=1.4,
            label=name,
            alpha=0.85,
        )

    ax.set_title("Feature distributions")
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.legend(loc="best", fontsize="small")
    return fig, ax
