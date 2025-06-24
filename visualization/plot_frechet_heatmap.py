import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic_2d

def plot_frechet_heatmap(df, x_col, color_col, frechet_col='Frechet_Distance', bins=50, save_path=None):
    df = df.copy()

    # Normalize Frechet to [0, 10]
    frechet_min = df[frechet_col].min()
    frechet_max = df[frechet_col].max()
    df['Frechet_Norm'] = 10 * (df[frechet_col] - frechet_min) / (frechet_max - frechet_min)

    # Log-transform x_col (add epsilon to avoid log(0))
    epsilon = 1e-6
    df['Log_' + x_col] = np.log(df[x_col] + epsilon)

    # Filter out inf/nan after log
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Log_' + x_col, 'Frechet_Norm', color_col])

    # Compute 2D binned average Trust
    stat, x_edges, y_edges, _ = binned_statistic_2d(
        x=df['Log_' + x_col],
        y=df['Frechet_Norm'],
        values=df[color_col],
        statistic='mean',
        bins=bins
    )

    # Plot
    plt.figure(figsize=(10, 6))
    extent = [x_edges.min(), x_edges.max(), y_edges.min(), y_edges.max()]
    plt.imshow(stat.T, origin='lower', extent=extent, aspect='auto', cmap='viridis')
    plt.colorbar(label=color_col)
    plt.xlabel(f'log({x_col})')
    plt.ylabel('Normalized Fréchet Distance (0–10)')
    plt.title(f'Mean {color_col} by log({x_col}) and Normalized Fréchet')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    plt.show()
