import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import os

def plot_trust_heatmap(df, var1, var2, var3, z_col='Expect', save_path=None):
    """
    Plot 3 interpolated trust surfaces using all pairs of var1, var2, var3.
    
    Parameters:
    - df (pd.DataFrame): The data.
    - var1, var2, var3 (str): Names of three independent variables.
    - z_col (str): Name of the trust variable (default: 'Trust').
    - save_path (str or None): Path to save the figure (optional).
    """
    df = df.copy()
    for col in [var1, var2, var3, z_col]:
        df[col] = df[col].astype(float)

    # Prepare variable combinations
    combinations = [
        (var1, var2),
        (var1, var3),
        (var2, var3)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 4))
    
    for ax, (x_col, y_col) in zip(axes, combinations):
        x = df[x_col]
        y = df[y_col]
        z = df[z_col]

        xi = np.linspace(x.min(), x.max(), 10000)
        yi = np.linspace(y.min(), y.max(), 10000)
        levels = np.linspace(z.min(), z.max(), 100)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = griddata((x, y), z, (Xi, Yi), method='cubic')

        contour = ax.contourf(Xi, Yi, Zi, levels=levels, cmap='viridis')
        ax.plot(x, y, 'o', markersize=2, color='black')
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f'{z_col} by {x_col} and {y_col}')
        ax.set_xlim(1, 10)
        ax.set_ylim(1, 10)
        ax.set_xticks(np.arange(1, 11, 1))
        ax.set_yticks(np.arange(1, 11, 1))
    
    # Add colorbar to the right of all subplots
    cbar = fig.colorbar(
        contour,
        ax=axes,
        location='right',
        shrink=1,
        aspect=30,
        pad=0.05,
        label = z_col
    )

    # plt.tight_layout()

    if save_path:
        if os.path.isdir(save_path):
            filename = f"{z_col}_heatmap.png"
            save_path = os.path.join(save_path, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    plt.show()
