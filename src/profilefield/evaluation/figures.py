"""Publication-ready diagnostic figures written as PNG, PDF, and editable SVG."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from profilefield.evaluation.metrics import correlation_from_covariance

FloatArray = NDArray[np.float64]

plt.rcParams["svg.fonttype"] = "none"


def _save_both(fig: plt.Figure, directory: Path, stem: str) -> None:
    fig.savefig(directory / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(directory / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def covariance_figure(
    true_covariance: FloatArray,
    estimates: dict[str, FloatArray],
    directory: Path,
    stem: str = "latent_correlation",
) -> None:
    matrices = {"Truth": correlation_from_covariance(true_covariance)}
    matrices.update({name: correlation_from_covariance(value) for name, value in estimates.items()})
    fig, axes = plt.subplots(1, len(matrices), figsize=(4 * len(matrices), 3.5), squeeze=False)
    image = None
    for axis, (name, matrix) in zip(axes[0], matrices.items(), strict=True):
        image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set_title(name)
        axis.set_xlabel("Latent output")
        axis.set_ylabel("Latent output")
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="Correlation")
    _save_both(fig, directory, stem)


def calibration_figure(rows: list[dict[str, Any]], directory: Path) -> None:
    frame = pd.DataFrame(rows)
    frame = frame[frame["output"].astype(str) == "all"]
    fig, axis = plt.subplots(figsize=(5.0, 4.5))
    axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="Ideal")
    for model, group in frame.groupby("model"):
        axis.plot(group["nominal_coverage"], group["empirical_coverage"], "o-", label=model)
    axis.set(
        xlabel="Nominal coverage", ylabel="Empirical coverage", xlim=(0.45, 1.0), ylim=(0.45, 1.0)
    )
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    _save_both(fig, directory, "calibration")


def field_figure(
    coordinates: FloatArray,
    truth: FloatArray,
    predictions: dict[str, FloatArray],
    posterior_draw: FloatArray,
    directory: Path,
) -> None:
    output = 0
    panels = [("Truth", truth[:, output])]
    panels.extend((name, values[:, output]) for name, values in predictions.items())
    panels.append(("LMC posterior draw", posterior_draw[:, output]))
    low = min(float(values.min()) for _, values in panels)
    high = max(float(values.max()) for _, values in panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.6), squeeze=False)
    image = None
    for axis, (name, values) in zip(axes[0], panels, strict=True):
        image = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=values,
            vmin=low,
            vmax=high,
            cmap="viridis",
            s=28,
        )
        axis.set_title(name)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.7)
    _save_both(fig, directory, "posterior_field_output0")


def variogram_figure(
    truth_rows: list[dict[str, Any]],
    estimates: dict[str, list[dict[str, Any]]],
    directory: Path,
) -> None:
    truth = pd.DataFrame(truth_rows)
    pairs = truth[["left_output", "right_output"]].drop_duplicates().head(4)
    fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 3.5), squeeze=False)
    for axis, pair in zip(axes[0], pairs.itertuples(index=False), strict=True):
        selected_truth = truth[
            (truth["left_output"] == pair.left_output)
            & (truth["right_output"] == pair.right_output)
        ]
        axis.plot(selected_truth["distance"], selected_truth["semivariance"], "ko-", label="Truth")
        for name, rows in estimates.items():
            frame = pd.DataFrame(rows)
            selected = frame[
                (frame["left_output"] == pair.left_output)
                & (frame["right_output"] == pair.right_output)
            ]
            axis.plot(selected["distance"], selected["semivariance"], "o-", label=name)
        axis.set_title(f"Outputs {pair.left_output}, {pair.right_output}")
        axis.set_xlabel("Lag distance")
        axis.set_ylabel("Cross-semivariance")
        axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    _save_both(fig, directory, "cross_variograms")
