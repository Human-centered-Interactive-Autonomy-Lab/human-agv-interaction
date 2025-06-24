import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd

def plot_agv_direction_trust(df, trust_col='Trust', direction_col='AGV_Approaching', save_path=None):
    """
    Draws a radial map showing average trust for each AGV approaching direction.
    """
    # Define all 8 directions (ordered clockwise)
    directions = ['North', 'Northeast', 'East', 'Southeast', 'South', 'Southwest', 'West', 'Northwest']
    angles = np.linspace(0, 360, len(directions) + 1)  # one extra to complete circle

    # Compute average trust per direction
    avg_trust = df.groupby(direction_col)[trust_col].mean().reindex(directions).fillna(0)

    # Normalize trust to 1–10
    norm = mcolors.Normalize(vmin=1, vmax=10)
    cmap = cm.Reds

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.axis('off')

    # Draw wedges
    for i, dir_name in enumerate(directions):
        start_angle = np.radians(angles[i])
        end_angle = np.radians(angles[i + 1])
        theta = (start_angle + end_angle) / 2
        color = cmap(norm(avg_trust[dir_name]))

        wedge = Wedge((0, 0), 1, np.degrees(start_angle), np.degrees(end_angle),
                      facecolor=color, edgecolor='white', linewidth=2)
        ax.add_patch(wedge)

        # Add direction label and trust value
        r = 1.2
        ax.text(theta, r, f"{dir_name}\n{avg_trust[dir_name]:.1f}", ha='center', va='center', fontsize=9)

    # Add user icon or placeholder at center
    ax.plot(0, 0, 'ko', markersize=10)
    ax.text(0, 0, "User", ha='center', va='center', fontsize=12, color='black')

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.1, shrink=0.7)
    cbar.set_label('Average Trust (1–10)')

    plt.title("Trust by AGV Approaching Direction", fontsize=14)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    plt.tight_layout()
    plt.show()
