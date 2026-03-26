import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch, Wedge
from matplotlib.colors import Normalize
import numpy as np
import plotly.graph_objects as go
from scipy.stats import mannwhitneyu
from typing import Dict, Tuple

# Set the default font family
plt.rcParams['font.family'] = 'serif'  # or 'sans-serif', 'monospace', 'cursive', 'fantasy'
plt.rcParams['font.serif'] = ['DejaVu Serif'] # 'Palatino', 'DejaVu Serif', Times New Roman # List of fonts to try
plt.rcParams['font.size'] = 15

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

def _pick_arc(gaze_deg, agv_deg, target_angle_deg, tol=2.0):
    """Return (theta1, theta2, arc_len_ccw) so that the CCW arc equals target_angle."""
    gaze_deg = gaze_deg % 360
    agv_deg  = agv_deg  % 360
    # Two possible arcs between directions:
    ccw_len = (agv_deg - gaze_deg + 360) % 360        # gaze -> agv (CCW)
    cw_len  = (gaze_deg - agv_deg + 360) % 360        # agv  -> gaze (CCW) == CW from gaze to agv

    # Choose the one that best matches target_angle
    if abs(ccw_len - target_angle_deg) <= abs(cw_len - target_angle_deg) + tol:
        # Draw CCW from gaze to agv
        return gaze_deg, agv_deg, ccw_len
    else:
        # Draw CCW from agv to gaze (i.e., CW from gaze to agv)
        return agv_deg, gaze_deg, cw_len

def plot_agv_user_fov(
    data,
    save_path = None,
    n_samples=6,
    fov_half_angle_deg=60,
    scale_factor=100.0,          # divide by this to get meters
    gaze_arrow_len_m=50,      # length of gaze/FOV vectors in meters
    random_state=2000,
    angle_col='Gaze_Angle_to_AGV',
    agv_in_fov_col='AGV_in_FOV',
    user_cols=('GazeOrigin_X', 'GazeOrigin_Y'),
    agv_cols=('AGV_X', 'AGV_Y'),
    gaze_cols=('GazeDirection_X', 'GazeDirection_Y'),
    xlim_m=(0, 300),
    ylim_m=(0, 300),
    annotate_angle=True,
):
    """
    Visualize random samples of user/AGV positions, gaze direction and FOV.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe with necessary columns.
    save_path : str or Path
        Full path (including filename) to save the resulting figure.
    n_samples : int
        Number of random rows to plot.
    fov_half_angle_deg : float
        Half of the field-of-view angle in degrees (e.g., 60 -> ±60° around gaze).
    scale_factor : float
        Divide (x, y) coordinates by this to convert to meters.
    gaze_arrow_len_m : float
        Length of the gaze/FOV arrows in meters.
    random_state : int
        Seed for reproducible sampling.
    angle_col : str
        Column name that stores the Angle to AGV (in degrees) to annotate.
    agv_in_fov_col : str
        Column name that indicates whether AGV is in FOV.
    user_cols, agv_cols, gaze_cols : tuple[str, str]
        Column names for user, AGV positions and gaze direction (x, y).
    xlim_m, ylim_m : tuple[float, float]
        Axis limits in meters.
    annotate_angle : bool
        If True, write the `angle_col` value between the user and AGV.

    Returns
    -------
    (fig, axes)
        Matplotlib figure and axes array.
    """
    sample_data = data.sample(n_samples, random_state=random_state).reset_index(drop=True)

    # Compute subplot grid automatically (here fixed to 2x4 for n_samples=8)
    nrows = 1
    ncols = min(4, n_samples)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)  # flatten for easy indexing

    half_rad = np.radians(fov_half_angle_deg)

    for idx, row in sample_data.iterrows():
        ax = axes[idx]

        # Positions in meters
        user_pos = np.array([row[user_cols[0]], row[user_cols[1]]], dtype=float) / scale_factor
        agv_pos  = np.array([row[agv_cols[0]],  row[agv_cols[1]]],  dtype=float) / scale_factor

        # Gaze (no scaling needed for direction)
        gaze_dir = np.array([row[gaze_cols[0]], row[gaze_cols[1]]], dtype=float)
        gaze_norm = gaze_dir / (np.linalg.norm(gaze_dir) + 1e-6)
        gaze_arrow = gaze_norm * gaze_arrow_len_m

        # Angles
        gaze_angle = np.arctan2(gaze_norm[1], gaze_norm[0])
        left_rad   = gaze_angle + half_rad
        right_rad  = gaze_angle - half_rad
        left_deg   = np.degrees(left_rad)
        right_deg  = np.degrees(right_rad)

        # FOV wedge
        wedge = Wedge(
            center=(user_pos[0], user_pos[1]),
            r=gaze_arrow_len_m,
            theta1=right_deg,
            theta2=left_deg,
            facecolor='green',
            alpha=0.2,
        )
        ax.add_patch(wedge)

        # FOV boundary vectors
        left_vec  = np.array([np.cos(left_rad),  np.sin(left_rad)])  * gaze_arrow_len_m
        right_vec = np.array([np.cos(right_rad), np.sin(right_rad)]) * gaze_arrow_len_m

        # Scatter user & AGV
        ax.scatter(*agv_pos,  color='red',  s=100, marker='^', label='AGV')
        ax.scatter(*user_pos, color='blue', s=100, marker='o', label='User Gaze Origin')

        # Gaze arrow
        # Start point
        x0, y0 = user_pos[0], user_pos[1]
        # End point = start + vector
        x1, y1 = x0 + gaze_arrow[0], y0 + gaze_arrow[1]

        ax.plot([x0, x1], [y0, y1],
                color='green', linewidth=2, linestyle='-',
                label='Gaze Direction')

        # FOV bounds
        ax.arrow(user_pos[0], user_pos[1], left_vec[0],  left_vec[1],
                 linestyle='dashed', color='blue', alpha=0.6)
        ax.arrow(user_pos[0], user_pos[1], right_vec[0], right_vec[1],
                 linestyle='dashed', color='blue', alpha=0.6)

        # Line user -> AGV
        agv_vec = agv_pos - user_pos
        agv_vec_norm = agv_vec / (np.linalg.norm(agv_vec) + 1e-6)
        agv_arrow = agv_vec_norm * gaze_arrow_len_m

        ax.plot(
            [user_pos[0], user_pos[0] + agv_arrow[0]],
            [user_pos[1], user_pos[1] + agv_arrow[1]],
            color='red',
            linestyle='--',
            label='AGV Direction'
        )

        # --- Red arc between gaze and AGV direction (match the given angle exactly) ---
        arc_radius = gaze_arrow_len_m * 0.6
        arc_label_radius = arc_radius + 5

        # Directions in degrees
        gaze_deg = (np.degrees(gaze_angle)) % 360
        agv_deg  = (np.degrees(np.arctan2(agv_vec_norm[1], agv_vec_norm[0]))) % 360
        target_angle = float(row[angle_col]) if not pd.isna(row[angle_col]) else np.nan

        if not pd.isna(target_angle):
            theta1, theta2, arc_len = _pick_arc(gaze_deg, agv_deg, target_angle, tol=2.0)

            # Draw the arc as a Wedge (outline+fill)
            arc_patch = Wedge(
                center=(user_pos[0], user_pos[1]),
                r=arc_radius,
                theta1=theta1,
                theta2=theta2,
                edgecolor='red',
                facecolor='red',
                alpha=0.2,
                linestyle='--',
                linewidth=2
            )
            ax.add_patch(arc_patch)

            # Label at the middle of the chosen arc
            mid_theta = (theta1 + arc_len / 2.0) % 360
            mid_theta_rad = np.radians(mid_theta)
            label_x = user_pos[0] + arc_label_radius * np.cos(mid_theta_rad)
            label_y = user_pos[1] + arc_label_radius * np.sin(mid_theta_rad)
            ax.text(label_x, label_y, f"{target_angle:.1f}°", color='red', ha='center', va='center')

        # Title
        # Retrieve values
        pid = int(row['PID'])
        agv_name = row['AGVname']  # e.g., 'AGV 3' or just '3'
        fov_status = str(row[agv_in_fov_col]).strip().lower()  # normalize string

        # Determine "in" or "not in"
        if 'false' in fov_status or fov_status in ['0', 'no', 'none']:
            visibility = "not in"
        else:
            visibility = "in"

        # Extract AGV number (if AGVname has a number in it)
        import re
        agv_number_match = re.search(r'\d+', str(agv_name))
        agv_number = agv_number_match.group(0) if agv_number_match else str(agv_name)

        # Final title
        title_str = f"AGV {visibility} User FOV"
        # title_str = f"AGV {agv_number} {visibility} User {pid} FOV"
        # ax.set_title(title_str)

        # Axes
        ax.set_xlim(*xlim_m)
        ax.set_ylim(*ylim_m)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_xlabel('X (m)')

        # Y label only for first and fifth axes
        if idx in [0, 4]:
            ax.set_ylabel('Y (m)')
        else:
            ax.set_ylabel('')

    # Legend once (last used axis)
    axes[min(n_samples, len(axes)) - 1].legend(loc='upper right')

    # Hide any extra axes if n_samples < nrows*ncols
    for j in range(n_samples, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.3, hspace=0.2) 

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    plt.show()

def plot_cluster_agreement_heatmap(
    behavioral_df,
    subjective_df,
    pid_col="PID",
    cluster_col="Cluster",
    save_path=None,
):
    beh = behavioral_df[[pid_col, cluster_col]].rename(
        columns={cluster_col: "Behavioral"}
    )
    sub = subjective_df[[pid_col, cluster_col]].rename(
        columns={cluster_col: "Subjective"}
    )

    merged = pd.merge(beh, sub, on=pid_col, how="inner")

    table = pd.crosstab(
        merged["Behavioral"],
        merged["Subjective"],
        rownames=["Behavioral Cluster"],
        colnames=["Subjective Cluster"],
    )

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        table,
        annot=True,
        fmt="d",
        cmap="Blues",
        linewidths=0.8,
        cbar=False,
    )

    plt.title("Cluster Agreement Between Behavioral and Subjective Methods")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_cluster_split_stacked_bars(
    behavioral_df,
    subjective_df,
    pid_col="PID",
    cluster_col="Cluster",
    save_path=None,
):
    beh = behavioral_df[[pid_col, cluster_col]].rename(
        columns={cluster_col: "Behavioral"}
    )
    sub = subjective_df[[pid_col, cluster_col]].rename(
        columns={cluster_col: "Subjective"}
    )

    merged = pd.merge(beh, sub, on=pid_col, how="inner")

    counts = (
        merged
        .groupby(["Behavioral", "Subjective"])
        .size()
        .unstack(fill_value=0)
    )

    colors = ["#1f77b4", "#ff7f0e"]  # blue / orange

    counts.plot(
        kind="bar",
        stacked=True,
        figsize=(6, 5),
        color=colors,
        edgecolor="black"
    )

    plt.xlabel("Behavioral Cluster")
    plt.ylabel("Number of PIDs")
    plt.title("Subjective Cluster Composition Within Behavioral Clusters")
    plt.legend(title="Subjective Cluster")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_agreement_group_counts(
    behavioral_df,
    subjective_df,
    pid_col="PID",
    cluster_col="Cluster",
    save_path=None,
):
    beh = behavioral_df[[pid_col, cluster_col]].rename(
        columns={cluster_col: "B"}
    )
    sub = subjective_df[[pid_col, cluster_col]].rename(
        columns={cluster_col: "S"}
    )

    merged = pd.merge(beh, sub, on=pid_col, how="inner")
    merged["Group"] = "B" + merged["B"].astype(str) + "_S" + merged["S"].astype(str)

    counts = merged["Group"].value_counts().sort_index()

    plt.figure(figsize=(6, 4))
    bars = plt.bar(counts.index, counts.values, color="#7f7f7f", edgecolor="black")

    plt.xlabel("Cluster Agreement Group")
    plt.ylabel("Number of PIDs")
    plt.title("PID Distribution Across Clustering Agreement Groups")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

    return counts

def plot_cluster_agreement_sankey(
    behavioral_df,
    subjective_df,
    pid_col="PID",
    cluster_col="Cluster",
    save_path=None,   # e.g., "agreement_sankey.html" or "agreement_sankey.png"
    title="Cluster Agreement Between Behavioral and Subjective Methods",
):
    """
    Sankey / alluvial diagram showing how participants flow from
    Behavioral clusters to Subjective clusters.

    Behavioral labels:
      - 1 -> Skeptical
      - 2 -> Deliberate

    Subjective labels:
      - 1 -> Low rating
      - 2 -> High rating
    """

    # --- Extract and merge ---
    beh = behavioral_df[[pid_col, cluster_col]].rename(columns={cluster_col: "Behavioral"})
    sub = subjective_df[[pid_col, cluster_col]].rename(columns={cluster_col: "Subjective"})
    merged = pd.merge(beh, sub, on=pid_col, how="inner").dropna(subset=["Behavioral", "Subjective"])

    # Optional: coerce to int if your clusters are numeric-but-stored-as-strings
    merged["Behavioral"] = pd.to_numeric(merged["Behavioral"], errors="ignore")
    merged["Subjective"] = pd.to_numeric(merged["Subjective"], errors="ignore")

    # --- Crosstab counts ---
    table = pd.crosstab(merged["Behavioral"], merged["Subjective"])

    # Stable ordering
    beh_clusters = sorted(table.index.tolist())
    sub_clusters = sorted(table.columns.tolist())

    # --- Custom label maps ---
    beh_name = {
        1: "Skeptical",
        2: "Deliberate",
    }
    sub_name = {
        1: "Lower Trust",
        2: "Higher Trust",
    }

    def _label_for_cluster(c, mapping, prefix):
        """
        c can be int/float/str depending on upstream; try to normalize to int if possible.
        """
        try:
            key = int(c)
        except Exception:
            key = c
        nice = mapping.get(key, f"C{c}")
        return f"{prefix}: {nice}"

    # Node labels
    beh_labels = [_label_for_cluster(c, beh_name, "Behavioral") for c in beh_clusters]
    sub_labels = [_label_for_cluster(c, sub_name, "Subjective") for c in sub_clusters]
    labels = beh_labels + sub_labels

    # Node index maps
    beh_idx = {c: i for i, c in enumerate(beh_clusters)}
    sub_idx = {c: i + len(beh_clusters) for i, c in enumerate(sub_clusters)}

    # Links
    sources, targets, values = [], [], []
    for b in beh_clusters:
        for s in sub_clusters:
            v = int(table.loc[b, s])
            if v > 0:
                sources.append(beh_idx[b])
                targets.append(sub_idx[s])
                values.append(v)

    # --- Build Sankey figure ---
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=18,
                    line=dict(width=0.5),
                    label=labels,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                ),
            )
        ]
    )

    fig.update_layout(
        # title_text=title,
        font=dict(
            family="DejaVu Serif, Times New Roman, serif",
            size=15,
            color="black",
        ),
        margin=dict(l=20, r=20, t=50, b=20),
    )


    # --- Save (optional) ---
    if save_path:
        sp = str(save_path).lower()
        if sp.endswith(".html"):
            fig.write_html(save_path)
        else:
            # For png/pdf/svg output, install kaleido:
            # pip install -U kaleido
            fig.write_image(save_path, scale=2)

    fig.show()

def get_cluster_label(cluster_id, cluster_file: pd.DataFrame):
    """
    Return a descriptive cluster label including the number of users.

    The label map is inferred from the source filename stored in
    cluster_file.attrs['source_name'] when the CSV is loaded.
    """
    count = (cluster_file['Cluster'] == cluster_id).sum()
    source_name = str(cluster_file.attrs.get('source_name', '')).lower()

    if 'behavioral' in source_name:
        label_map = {
            1: 'Skeptical',
            2: 'Deliberate',
        }
    elif 'subjective' in source_name:
        label_map = {
            1: 'Lower Trust',
            2: 'Higher Trust',
        }
    else:
        label_map = {}

    label = label_map.get(cluster_id, f"Cluster {cluster_id}")
    return f"{label} (N={count})"


def _p_to_math_text(p: float) -> str:
    if p < 0.001:
        return r"$p < 0.001$"
    if p < 0.01:
        return r"$p < 0.01$"
    if p < 0.05:
        return r"$p < 0.05$"
    return ""


def _prepare_cross_first_between_clusters(
    df: pd.DataFrame,
    cluster_file: pd.DataFrame,
    *,
    id_col: str,
    cluster_col: str,
    decision_col: str,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if id_col not in df.columns:
        raise ValueError(f"{id_col} not found in df. Available: {df.columns.tolist()}")
    if decision_col not in df.columns:
        raise ValueError(f"{decision_col} not found in df. Available: {df.columns.tolist()}")
    if id_col not in cluster_file.columns:
        raise ValueError(f"{id_col} not found in cluster_file. Available: {cluster_file.columns.tolist()}")
    if cluster_col not in cluster_file.columns:
        raise ValueError(f"{cluster_col} not found in cluster_file. Available: {cluster_file.columns.tolist()}")

    merged = (
        pd.merge(
            df[[id_col, decision_col]].copy(),
            cluster_file[[id_col, cluster_col]].copy(),
            on=id_col,
            how="left",
        )
        .dropna(subset=[cluster_col])
        .copy()
    )

    merged[cluster_col] = merged[cluster_col].astype(str)

    def to_collapsed(val) -> str:
        if pd.isna(val):
            return "N/A"
        value = str(val).strip().lower()
        if value == "user":
            return "User"
        if value == "agv":
            return "AGV"
        return "N/A"

    categories = ["User", "AGV", "N/A"]
    merged["Cross_First_collapsed"] = merged[decision_col].map(to_collapsed)

    pid_counts = (
        merged.groupby([id_col, cluster_col, "Cross_First_collapsed"])
        .size()
        .unstack("Cross_First_collapsed", fill_value=0)
        .reset_index()
    )

    for cat in categories:
        if cat not in pid_counts.columns:
            pid_counts[cat] = 0

    totals = pid_counts[categories].sum(axis=1).replace(0, np.nan)
    pid_props = pid_counts[[id_col, cluster_col]].copy()
    for cat in categories:
        pid_props[cat] = (pid_counts[cat] / totals) * 100.0
    pid_props[categories] = pid_props[categories].fillna(0.0)

    clusters = sorted(pid_props[cluster_col].unique().tolist())
    if len(clusters) != 2:
        raise ValueError(f"Expected exactly 2 clusters, found {len(clusters)}: {clusters}")

    c1, c2 = clusters[0], clusters[1]
    pvals: Dict[str, float] = {}
    for cat in categories:
        a = pid_props.loc[pid_props[cluster_col] == c1, cat].to_numpy(float)
        b = pid_props.loc[pid_props[cluster_col] == c2, cat].to_numpy(float)
        if len(a) == 0 or len(b) == 0:
            pvals[cat] = np.nan
        else:
            pvals[cat] = mannwhitneyu(a, b, alternative="two-sided").pvalue

    return pid_props, pvals


def _plot_cross_first_between_clusters_axis(
    ax,
    pid_props: pd.DataFrame,
    pvals: Dict[str, float],
    cluster_file: pd.DataFrame,
    *,
    cluster_col: str,
    colors: Tuple[str, str],
    title: str,
    show_ylabel: bool,
):
    categories = ["User", "AGV", "N/A"]
    category_display = {"User": "User", "AGV": "AGV", "N/A": "No Interaction"}
    hatches_c1 = ["o", "o-", "ooo"]
    hatches_c2 = ["/", "\\", "||"]

    clusters = sorted(pid_props[cluster_col].unique().tolist())
    c1, c2 = clusters[0], clusters[1]
    means_c1 = [pid_props.loc[pid_props[cluster_col] == c1, cat].mean() for cat in categories]
    means_c2 = [pid_props.loc[pid_props[cluster_col] == c2, cat].mean() for cat in categories]

    x = np.arange(len(categories))
    bar_w = 0.42
    x_c1 = x - bar_w / 2
    x_c2 = x + bar_w / 2

    bars1 = ax.bar(x_c1, means_c1, width=bar_w, color=colors[0], edgecolor="black", linewidth=1.0)
    for bar, hatch in zip(bars1, hatches_c1):
        bar.set_hatch(hatch)

    bars2 = ax.bar(x_c2, means_c2, width=bar_w, color=colors[1], edgecolor="black", linewidth=1.0)
    for bar, hatch in zip(bars2, hatches_c2):
        bar.set_hatch(hatch)

    ax.set_title(title)
    ax.set_xlabel("Cross First", fontsize=14, fontfamily='serif')
    ax.set_xticks(x)
    ax.set_xticklabels([category_display[c] for c in categories], rotation=0)
    if show_ylabel:
        ax.set_ylabel("Percentage (%)", fontsize=14, fontfamily='serif')

    ymax = 100
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.xaxis.grid(False)
    ax.tick_params(axis='both', labelsize=12)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontfamily('serif')

    def add_bracket(x_left, x_right, y_base, text, height=2, text_pad=1.2, lw=1.5):
        y_top = y_base + height
        ax.plot([x_left, x_left, x_right, x_right], [y_base, y_top, y_top, y_base], color="black", linewidth=lw)
        ax.text((x_left + x_right) / 2, y_top + text_pad, text, ha="center", va="bottom", fontsize=14)

    for i, cat in enumerate(categories):
        p = pvals.get(cat, np.nan)
        if not np.isfinite(p):
            continue
        p_label = _p_to_math_text(p)
        if not p_label:
            continue
        y_here = max(means_c1[i], means_c2[i])
        y_base = min(y_here + 3, ymax - 8)
        add_bracket(x_c1[i], x_c2[i], y_base, p_label)

    ax.legend(
        handles=[
            Patch(facecolor=colors[0], edgecolor="black", hatch="o", label=get_cluster_label(int(c1), cluster_file)),
            Patch(facecolor=colors[1], edgecolor="black", hatch="/", label=get_cluster_label(int(c2), cluster_file)),
        ],
        loc="upper left",
        fontsize=12,
        prop={'family': 'serif'},
    )


def plot_cross_first_cross_cluster(
    df: pd.DataFrame,
    behavioral_cluster_file: pd.DataFrame,
    subjective_cluster_file: pd.DataFrame,
    *,
    id_col: str = "PID",
    cluster_col: str = "Cluster",
    decision_col: str = "Cross_First",
    figsize: Tuple[int, int] = (16, 5),
    save_path: str | None = None,
):
    """
    Plot cross-first distributions side by side for behavioral and subjective clustering,
    sharing a common y-axis.

    Returns:
      - behavioral_pid_props, behavioral_pvals, subjective_pid_props, subjective_pvals
    """
    behavioral_pid_props, behavioral_pvals = _prepare_cross_first_between_clusters(
        df,
        behavioral_cluster_file,
        id_col=id_col,
        cluster_col=cluster_col,
        decision_col=decision_col,
    )
    subjective_pid_props, subjective_pvals = _prepare_cross_first_between_clusters(
        df,
        subjective_cluster_file,
        id_col=id_col,
        cluster_col=cluster_col,
        decision_col=decision_col,
    )

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=True, sharey=True)

    _plot_cross_first_between_clusters_axis(
        axes[0],
        behavioral_pid_props,
        behavioral_pvals,
        behavioral_cluster_file,
        cluster_col=cluster_col,
        colors=("#1f77b4", "#ff7f0e"),
        title="Behavioral Clustering",
        show_ylabel=True,
    )
    _plot_cross_first_between_clusters_axis(
        axes[1],
        subjective_pid_props,
        subjective_pvals,
        subjective_cluster_file,
        cluster_col=cluster_col,
        colors=("#6a3d9a", "#2ca02c"),
        title="Subjective Clustering",
        show_ylabel=False,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

    return behavioral_pid_props, behavioral_pvals, subjective_pid_props, subjective_pvals


def plot_efficiency_cross_cluster(
    data: pd.DataFrame,
    behavioral_cluster_file: pd.DataFrame,
    subjective_cluster_file: pd.DataFrame,
    save_path=None,
):
    """
    Plot trial time (efficiency) side by side comparing behavioral vs subjective clustering.
    Also exports each subplot individually.
    
    Parameters:
    - data: DataFrame with interaction-level data (PID, Interaction_No, StartTime, EndTime, etc.)
    - behavioral_cluster_file: DataFrame with (PID, Cluster) for behavioral clustering
    - subjective_cluster_file: DataFrame with (PID, Cluster) for subjective clustering
    - save_path: Optional base path to save the figures (will create _behavioral.png and _subjective.png)
    """
    
    # Compute interaction time
    df = data.copy()
    if 'StartTime' in df.columns and 'EndTime' in df.columns:
        df['Interaction_Time'] = (
            pd.to_datetime(df['EndTime']) - pd.to_datetime(df['StartTime'])
        ).dt.total_seconds()
    else:
        raise ValueError("Data must contain 'StartTime' and 'EndTime' columns")
    
    # Merge with both cluster assignments
    df = df.merge(behavioral_cluster_file[['PID', 'Cluster']], on='PID', how='inner', suffixes=('', '_behavioral'))
    df.rename(columns={'Cluster': 'Behavioral_Cluster'}, inplace=True)
    df = df.merge(subjective_cluster_file[['PID', 'Cluster']], on='PID', how='inner')
    df.rename(columns={'Cluster': 'Subjective_Cluster'}, inplace=True)
    
    # Color schemes
    behavioral_colors = {1: "#1f77b4", 2: "#ff7f0e"}  # blue, orange
    subjective_colors = {1: "#6a3d9a", 2: "#2ca02c"}  # purple, green
    
    interaction_range = sorted(df['Interaction_No'].dropna().unique())
    
    # --- Individual plot: Behavioral clustering ---
    fig_beh, ax_beh = plt.subplots(1, 1, figsize=(8, 4))
    
    for cluster_id in sorted(df['Behavioral_Cluster'].dropna().unique()):
        cluster_id_int = int(cluster_id)
        sub = df[df['Behavioral_Cluster'] == cluster_id]
        
        agg = (
            sub.groupby('Interaction_No')['Interaction_Time']
            .agg(['mean', 'count', 'std'])
            .reindex(interaction_range)
        )
        agg['sem'] = agg['std'] / np.sqrt(agg['count'].clip(lower=1))
        
        c = behavioral_colors.get(cluster_id_int, "0.4")
        ls = '-' if cluster_id_int == 1 else '--'
        
        ax_beh.errorbar(
            agg.index, agg['mean'], yerr=agg['sem'],
            capsize=4, marker='o', linestyle=ls,
            color=c, ecolor=c,
            label=get_cluster_label(cluster_id_int, behavioral_cluster_file)
        )
    
    ax_beh.set_xlabel("Interaction Number", fontsize=14, fontfamily='serif')
    ax_beh.set_ylabel("Trial Time (s)", fontsize=14, fontfamily='serif')
    ax_beh.grid(True, linestyle='--', alpha=0.3)
    ax_beh.legend(loc='upper right', fontsize=12, prop={'family': 'serif'})
    ax_beh.tick_params(axis='both', labelsize=12)
    for t in ax_beh.get_xticklabels() + ax_beh.get_yticklabels():
        t.set_fontfamily('serif')
    
    plt.tight_layout()
    if save_path:
        base_path = str(save_path).rsplit('.', 1)[0]
        ext = str(save_path).rsplit('.', 1)[1] if '.' in str(save_path) else 'png'
        behavioral_path = f"{base_path}_behavioral.{ext}"
        plt.savefig(behavioral_path, bbox_inches='tight', dpi=300)
    plt.close(fig_beh)
    
    # --- Individual plot: Subjective clustering ---
    fig_sub, ax_sub = plt.subplots(1, 1, figsize=(8, 4))
    
    for cluster_id in sorted(df['Subjective_Cluster'].dropna().unique()):
        cluster_id_int = int(cluster_id)
        sub = df[df['Subjective_Cluster'] == cluster_id]
        
        agg = (
            sub.groupby('Interaction_No')['Interaction_Time']
            .agg(['mean', 'count', 'std'])
            .reindex(interaction_range)
        )
        agg['sem'] = agg['std'] / np.sqrt(agg['count'].clip(lower=1))
        
        c = subjective_colors.get(cluster_id_int, "0.4")
        ls = '-' if cluster_id_int == 1 else '--'
        
        ax_sub.errorbar(
            agg.index, agg['mean'], yerr=agg['sem'],
            capsize=4, marker='o', linestyle=ls,
            color=c, ecolor=c,
            label=get_cluster_label(cluster_id_int, subjective_cluster_file)
        )
    
    ax_sub.set_xlabel("Interaction Number", fontsize=14, fontfamily='serif')
    ax_sub.set_ylabel("Trial Time (s)", fontsize=14, fontfamily='serif')
    ax_sub.grid(True, linestyle='--', alpha=0.3)
    ax_sub.legend(loc='upper right', fontsize=12, prop={'family': 'serif'})
    ax_sub.tick_params(axis='both', labelsize=12)
    for t in ax_sub.get_xticklabels() + ax_sub.get_yticklabels():
        t.set_fontfamily('serif')
    
    plt.tight_layout()
    if save_path:
        base_path = str(save_path).rsplit('.', 1)[0]
        ext = str(save_path).rsplit('.', 1)[1] if '.' in str(save_path) else 'png'
        subjective_path = f"{base_path}_subjective.{ext}"
        plt.savefig(subjective_path, bbox_inches='tight', dpi=300)
    plt.close(fig_sub)
    
    # --- Combined side-by-side plot ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharex=True, sharey=True)
    
    # Left plot: Behavioral clustering
    ax = axes[0]
    for cluster_id in sorted(df['Behavioral_Cluster'].dropna().unique()):
        cluster_id_int = int(cluster_id)
        sub = df[df['Behavioral_Cluster'] == cluster_id]
        
        agg = (
            sub.groupby('Interaction_No')['Interaction_Time']
            .agg(['mean', 'count', 'std'])
            .reindex(interaction_range)
        )
        agg['sem'] = agg['std'] / np.sqrt(agg['count'].clip(lower=1))
        
        c = behavioral_colors.get(cluster_id_int, "0.4")
        ls = '-' if cluster_id_int == 1 else '--'
        
        ax.errorbar(
            agg.index, agg['mean'], yerr=agg['sem'],
            capsize=4, marker='o', linestyle=ls,
            color=c, ecolor=c,
            label=get_cluster_label(cluster_id_int, behavioral_cluster_file)
        )
    
    ax.set_xlabel("Interaction Number", fontsize=14, fontfamily='serif')
    ax.set_ylabel("Trial Time (s)", fontsize=14, fontfamily='serif')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=12, prop={'family': 'serif'})
    ax.tick_params(axis='both', labelsize=12)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontfamily('serif')
    
    # Right plot: Subjective clustering
    ax = axes[1]
    for cluster_id in sorted(df['Subjective_Cluster'].dropna().unique()):
        cluster_id_int = int(cluster_id)
        sub = df[df['Subjective_Cluster'] == cluster_id]
        
        agg = (
            sub.groupby('Interaction_No')['Interaction_Time']
            .agg(['mean', 'count', 'std'])
            .reindex(interaction_range)
        )
        agg['sem'] = agg['std'] / np.sqrt(agg['count'].clip(lower=1))
        
        c = subjective_colors.get(cluster_id_int, "0.4")
        ls = '-' if cluster_id_int == 1 else '--'
        
        ax.errorbar(
            agg.index, agg['mean'], yerr=agg['sem'],
            capsize=4, marker='o', linestyle=ls,
            color=c, ecolor=c,
            label=get_cluster_label(cluster_id_int, subjective_cluster_file)
        )
    
    ax.set_xlabel("Interaction Number", fontsize=14, fontfamily='serif')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=12, prop={'family': 'serif'})
    ax.tick_params(axis='both', labelsize=12)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontfamily('serif')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
