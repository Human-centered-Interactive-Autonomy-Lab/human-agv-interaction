import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os

def plot_trust_transition(df, save_path=None):
    df = df[df['Trust_before'].notna()]

    unique_agv = df['AGV_Approaching'].unique()
    unique_traj = df['User_Trajectory'].unique()

    # Initialize nested transition matrices
    transition_matrices = {
        agv: {traj: np.zeros((10, 10)) for traj in unique_traj}
        for agv in unique_agv
    }

    # Fill matrices
    for _, row in df.iterrows():
        agv = row['AGV_Approaching']
        traj = row['User_Trajectory']
        trust_before = int(row['Trust_before']) - 1
        trust_after = int(row['Trust']) - 1
        if 0 <= trust_before < 10 and 0 <= trust_after < 10:
            transition_matrices[agv][traj][trust_before, trust_after] += 1

    fig, axes = plt.subplots(len(unique_traj), len(unique_agv), figsize=(5 * len(unique_agv), 6 * len(unique_traj)))
    cbar_ax = fig.add_axes([0.92, 0.2, 0.01, 0.6])
    last_img = None  # for shared colorbar

    for i, traj in enumerate(unique_traj):
        for j, agv in enumerate(unique_agv):
            ax = axes[i, j]
            matrix = transition_matrices[agv][traj]
            size = matrix.shape[0]

            # Diagonal line
            ax.plot([0.5, size - 0.5], [0.5, size - 0.5], color='blue', linestyle='-', linewidth=1.5)

            # Scatter and regression (weighted)
            X, Y = np.meshgrid(np.arange(size), np.arange(size))
            values = matrix.flatten()
            mask = values > 0
            if mask.sum() > 1:
                X_vals = X.flatten()[mask]
                Y_vals = Y.flatten()[mask]
                weights = values[mask]

                coeffs = np.polyfit(X_vals, Y_vals, 1, w=weights)
                y_reg = np.poly1d(coeffs)(np.arange(size))
                ax.scatter(X_vals + 1, Y_vals + 1, s=weights * 5, color='green', alpha=0.6)

            # Arrow vectors
            for row_idx in range(matrix.shape[0]):
                row_values = matrix[row_idx]
                total = row_values.sum()
                if total > 0:
                    avg = np.dot(np.arange(10), row_values) / total
                    ax.arrow(row_idx + 1, row_idx + 1, 0, avg - row_idx, head_width=0.3, head_length=0.5, fc='blue', ec='blue')

            # Center of mass curve
            center_of_mass = []
            for row_idx in range(matrix.shape[0]):
                total = matrix[row_idx].sum()
                if total > 0:
                    weighted_avg = np.dot(np.arange(10), matrix[row_idx]) / total
                    center_of_mass.append(weighted_avg)
                else:
                    center_of_mass.append(np.nan)
            ax.plot(np.arange(1, 11), np.array(center_of_mass) + 1, color='red', linestyle='--', linewidth=2)

            # Heatmap
            img = ax.imshow(matrix, cmap='binary', norm=Normalize(vmin=1, vmax=10),
                            extent=[0.5, 10.5, 10.5, 0.5])
            last_img = img  # for colorbar

            # Formatting
            ax.set_xticks(np.arange(1, 11))
            ax.set_yticks(np.arange(1, 11))
            
            ax.set_xlim(0.5, 10.5)
            ax.set_ylim(10.5, 0.5)

            if i == 0:
                ax.set_title(agv, fontsize=12)
            if j == 0:
                ax.set_ylabel(traj, fontsize=12)
            if i != len(unique_traj) - 1:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])
            ax.invert_yaxis()
    # Add shared colorbar
    if last_img is not None:
        fig.colorbar(last_img, cax=cbar_ax, shrink=1, pad=0.2)

    plt.subplots_adjust(left=0.06, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.4)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    plt.show()
