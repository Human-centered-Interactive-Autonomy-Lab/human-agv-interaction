import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os

def plot_3d_response_surface(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    title: str = '',
    elev: int = 45,
    azim: int = 135,
    method: str = 'nearest',
    figsize: tuple = (6, 5),
    save_path: str = None,
):
    """
    Plots a 3D surface for given x, y, and z column names in a DataFrame.
    
    Parameters:
    - df: pandas.DataFrame
    - x_col, y_col, z_col: str, column names for x, y, z axes
    - title: str, plot title
    - elev, azim: int, elevation and azimuth angles for 3D view
    - method: str, interpolation method ('linear', 'cubic', 'nearest')
    - figsize: tuple, figure size
    - save_path: str or None, either a directory or a full file path
    """
    # Drop rows with NaNs in any of the selected columns
    df_filtered = df[[x_col, y_col, z_col]].dropna()

    if df_filtered.empty:
        raise ValueError("No valid data available after dropping NaNs.")

    x = df_filtered[x_col].values
    y = df_filtered[y_col].values
    z = df_filtered[z_col].values

    # Generate grid
    xi = np.linspace(np.min(x), np.max(x), 15)
    yi = np.linspace(np.min(y), np.max(y), 15)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate Z values on grid
    zi = griddata((x, y), z, (xi, yi), method=method)

    zmin, zmax = np.nanmin(z), np.nanmax(z)
    zi = np.clip(zi, zmin, zmax)

    # Plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi, yi, zi, cmap='Blues', edgecolor='k', linewidth=0.5, alpha=0.9)

    ax.set_xlabel(x_col.replace('_', ' '))
    ax.set_ylabel(y_col.replace('_', ' '))
    ax.set_zlabel(z_col.replace('_', ' '))
    ax.set_title(title or f"{z_col.replace('_', ' ')} Surface Plot")
    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()

    if save_path:
        if os.path.isdir(save_path):
            filename = f"3d_surface_{z_col}.png"
            save_path = os.path.join(save_path, filename)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()
