from pathlib import Path
import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set the default font family
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams["font.size"] = 20

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def run_feature_diagnostics(
    data_path: Path,
    feature_cols: list[str],
    method: str = "pearson",
    triangle: str = "upper",
    bubble_cmap: str = "RdBu_r",          # kept for compatibility; used as cell colormap
    bubble_size_scale: float = 2200.0,    # kept for compatibility; unused now
    show: bool = True,
    save_corr_csv_path: Path | None = None,
    save_fig_path: Path | None = None,
    highlight_diagonal_features: list[str] | None = None,
):
    """
    Compute a correlation matrix and plot a half-triangle correlation table
    with numeric correlation values in each cell.

    - Cells are colored by correlation value using bubble_cmap.
    - Values are annotated (e.g., 0.73, -0.12).
    - Optionally highlight diagonal squares for selected features.
    - Saves CSV only if save_corr_csv_path is provided.
    - Saves figure only if save_fig_path is provided.
    """

    if triangle not in {"upper", "lower"}:
        raise ValueError("triangle must be 'upper' or 'lower'")

    logger.info("Loading data from %s", data_path)
    df = pd.read_csv(data_path)

    logger.info("Computing %s correlation matrix (%d features)", method, len(feature_cols))
    X = df[feature_cols]
    corr = X.corr(method=method)

    if save_corr_csv_path is not None:
        save_corr_csv_path = Path(save_corr_csv_path)
        save_corr_csv_path.parent.mkdir(parents=True, exist_ok=True)
        corr.to_csv(save_corr_csv_path)
        logger.info("Saved correlation matrix to %s", save_corr_csv_path)

    pretty_labels = [c.replace("_", " ") for c in feature_cols]
    n = len(feature_cols)

    # ---- BIG FIGURE SIZE ----
    fig_size = max(10, 1.2 * n)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Mask for triangle
    mask = np.ones((n, n), dtype=bool)
    if triangle == "upper":
        mask[np.triu_indices(n, k=0)] = False   # show upper incl. diagonal
    else:
        mask[np.tril_indices(n, k=0)] = False   # show lower incl. diagonal

    # Convert to masked array for plotting
    data = corr.values.copy()
    data_masked = np.ma.array(data, mask=mask)

    # Plot colored cells (no seaborn)
    im = ax.imshow(data_masked, vmin=-1, vmax=1, cmap=bubble_cmap)

    # Draw grid boxes only where cells are shown
    for i in range(n):
        for j in range(n):
            keep = (j >= i) if triangle == "upper" else (i >= j)
            if keep:
                ax.add_patch(
                    patches.Rectangle(
                        (j - 0.5, i - 0.5),
                        1, 1,
                        fill=False,
                        linewidth=1.0,
                        edgecolor="0.8"
                    )
                )

    # Annotate numeric correlation values
    # (choose text color based on background intensity)
    for i in range(n):
        for j in range(n):
            keep = (j >= i) if triangle == "upper" else (i >= j)
            if not keep:
                continue
            r = data[i, j]
            txt_color = "white" if abs(r) >= 0.5 else "black"
            ax.text(
                j, i,
                f"{r:.2f}",
                ha="center", va="center",
                color=txt_color,
                fontsize=20,
            )

    # Highlight selected diagonal squares
    if highlight_diagonal_features is not None:
        feature_to_idx = {f: i for i, f in enumerate(feature_cols)}
        for feat in highlight_diagonal_features:
            if feat not in feature_to_idx:
                logger.warning("Feature '%s' not found in feature_cols; skipping highlight.", feat)
                continue
            i = feature_to_idx[feat]
            ax.add_patch(
                patches.Rectangle(
                    (i - 0.5, i - 0.5),
                    1, 1,
                    fill=False,
                    linewidth=3.5,
                    edgecolor="blue",
                    linestyle="--",
                )
            )

    # Axes formatting
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(pretty_labels, rotation=90, ha="center", va="top")
    ax.set_yticklabels(pretty_labels, rotation=0)

    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"{method.capitalize()} Correlation Matrix", pad=25)

    # Colorbar sizing controls
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02, shrink=0.9)
    cbar.set_ticks(np.linspace(-1, 1, 11))

    plt.tight_layout()

    # Save figure only if requested
    if save_fig_path is not None:
        fig_out = save_fig_path / "subjective_correlation_heatmap.png"
        plt.savefig(fig_out, dpi=300, bbox_inches="tight")
        logger.info("Saved correlation heatmap to %s", fig_out)

    if show:
        plt.show()

    plt.close(fig)
    return corr

def _draw_corr_numbers_on_ax(
    ax,
    corr,                       # pd.DataFrame
    labels,                     # list[str] (same order as corr)
    triangle="upper",           # "upper" or "lower"
    cmap="RdBu_r",
    value_fontsize=14,
    title=None,
    highlight_diagonal_features=None,  # list[str] or None
):
    if triangle not in {"upper", "lower"}:
        raise ValueError("triangle must be 'upper' or 'lower'")

    n = len(labels)
    data = corr.values.astype(float)

    # Mask to show only chosen triangle (incl diagonal)
    mask = np.ones((n, n), dtype=bool)
    if triangle == "upper":
        mask[np.triu_indices(n, k=0)] = False
    else:
        mask[np.tril_indices(n, k=0)] = False

    data_masked = np.ma.array(data, mask=mask)

    # Colored cells
    im = ax.imshow(data_masked, vmin=-1, vmax=1, cmap=cmap)

    # Grid boxes only where shown
    for i in range(n):
        for j in range(n):
            keep = (j >= i) if triangle == "upper" else (i >= j)
            if keep:
                ax.add_patch(
                    patches.Rectangle(
                        (j - 0.5, i - 0.5), 1, 1,
                        fill=False, linewidth=1.0, edgecolor="0.8"
                    )
                )

    # Numeric annotations
    for i in range(n):
        for j in range(n):
            keep = (j >= i) if triangle == "upper" else (i >= j)
            if not keep:
                continue
            r = data[i, j]
            txt_color = "white" if abs(r) >= 0.5 else "black"
            ax.text(
                j, i, f"{r:.2f}",
                ha="center", va="center",
                color=txt_color,
                fontsize=value_fontsize,
            )

    # Optional: red outline on selected diagonal features
    if consider := highlight_diagonal_features:
        label_to_idx = {lab: idx for idx, lab in enumerate(labels)}
        for feat in consider:
            if feat not in label_to_idx:
                continue
            i = label_to_idx[feat]
            ax.add_patch(
                patches.Rectangle(
                    (i - 0.5, i - 0.5), 1, 1,
                    fill=False, linewidth=3.0, edgecolor="red"
                )
            )

    # Axis formatting
    pretty = [c.replace("_", " ") for c in labels]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(pretty, rotation=90, ha="center", va="top")
    ax.set_yticklabels(pretty, rotation=0)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_aspect("equal")

    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    if title:
        ax.set_title(title, pad=12)

    return im

def plot_2x2_subjective_corr_matrices(
    data_path: Path,
    method: str = "pearson",
    triangle: str = "upper",
    cmap: str = "RdBu_r",
    value_fontsize: int = 14,
    show: bool = True,
    save_fig_path: Path | None = None,
):
    df = pd.read_csv(data_path)

    sets = [
        ("Trust", trust_cols),
        ("Safe", safe_cols),
        ("Comfort", comfort_cols),
        ("Expect", expect_cols),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 20), constrained_layout=True)
    axes = axes.ravel()

    last_im = None
    for ax, (title, cols) in zip(axes, sets):
        X = df[cols]
        corr = X.corr(method=method)
        last_im = _draw_corr_numbers_on_ax(
            ax=ax,
            corr=corr,
            labels=cols,
            triangle=triangle,
            cmap=cmap,
            value_fontsize=value_fontsize,
            title=title,
        )

    # one shared colorbar for the whole 2×2 figure
    cbar = fig.colorbar(last_im, ax=axes, fraction=0.03, pad=0.02, shrink=0.9)
    cbar.set_ticks(np.linspace(-1, 1, 11))

    if save_fig_path is not None:
        fig_out = save_fig_path / "subjective_corr_2x2.png"
        plt.savefig(fig_out, dpi=300, bbox_inches="tight")
        logger.info("Saved correlation heatmap to %s", fig_out)

    if show:
        plt.show()
    plt.close(fig)

processed_data_dir = os.path.join(os.path.join(os.getcwd(), "data"), "processed")
processed_dir = Path(processed_data_dir)

trust_cols = [
    "Trust",
    "Trust_Mean", "Trust_Slope", "Trust_STD", "Trust_Range",
    "Trust_Mode", "Trust_Median", "Trust_Skewness",
]

safe_cols = [
    "Safe",
    "Safe_Mean", "Safe_Slope", "Safe_STD", "Safe_Range",
    "Safe_Mode", "Safe_Median", "Safe_Skewness",
]

comfort_cols = [
    "Comfort",
    "Comfort_Mean", "Comfort_Slope", "Comfort_STD", "Comfort_Range",
    "Comfort_Mode", "Comfort_Median", "Comfort_Skewness",
]

expect_cols = [
    "Expect",
    "Expect_Mean", "Expect_Slope", "Expect_STD", "Expect_Range",
    "Expect_Mode", "Expect_Median", "Expect_Skewness",
]

feature_cols = [
    "Trust", "Safe", "Expect", "Comfort",
    "Trust_Diff_Before", "Trust_Diff_After",
    "Expect_Diff_Before", "Expect_Diff_After",
    "Safe_Diff_Before", "Safe_Diff_After",
    "Comfort_Diff_Before", "Comfort_Diff_After"
    ]

corr = run_feature_diagnostics(
    data_path=processed_dir / "subjective_clustering_data.csv",
    feature_cols=feature_cols,
    method="kendall",
    triangle="upper",
    show=True,
    save_corr_csv_path=None,
    save_fig_path=Path.cwd() / "clustering" / "subjective_trust" /"figures" / "analysis" / "correlations",
    highlight_diagonal_features = [
    "Trust",
    "Trust_Diff_Before", "Trust_Diff_After",
    "Expect_Diff_Before", "Expect_Diff_After",
    "Safe_Diff_Before", "Safe_Diff_After",
    "Comfort_Diff_Before", "Comfort_Diff_After"
    ]
)

'''
plot_2x2_subjective_corr_matrices(
    data_path=processed_dir / "subjective_clustering_data.csv",  # or your subjective dataset path
    method="kendall",
    triangle="upper",
    cmap="RdBu_r",
    value_fontsize=14,
    save_fig_path=None, #Path.cwd() / "clustering" / "subjective_trust" /"figures" / "analysis" / "correlations",
)
'''