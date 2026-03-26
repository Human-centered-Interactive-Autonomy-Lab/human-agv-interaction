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
    fig_size = max(10, 2.2 * n)
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
                fontsize=25,
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
        fig_out = save_fig_path / "behavioral_correlation_heatmap.png"
        plt.savefig(fig_out, dpi=300, bbox_inches="tight")
        logger.info("Saved correlation heatmap to %s", fig_out)

    if show:
        plt.show()

    plt.close(fig)
    return corr


processed_data_dir = os.path.join(os.path.join(os.getcwd(), "data"), "raw")
processed_dir = Path(processed_data_dir)

feature_cols = [
    'Frechet_Distance', 
    'Gaze_Angle_to_AGV', 
    'User_Speed', 
    'Gaze_Instability',
    'Time_to_Collision',
    'Cumulative_Gaze_on_AGV'
]

corr = run_feature_diagnostics(
    data_path=processed_dir / "per_second_data.csv",
    feature_cols=feature_cols,
    method="pearson",
    triangle="upper",
    show=True,
    save_corr_csv_path=None,
    save_fig_path=Path.cwd() / "clustering" / "behavioral_trust" /"figures" / "analysis"/ "correlations",
    highlight_diagonal_features=[
        'Frechet_Distance',
        'Gaze_Angle_to_AGV',
        'User_Speed',
        'Gaze_Instability',
    ]
)