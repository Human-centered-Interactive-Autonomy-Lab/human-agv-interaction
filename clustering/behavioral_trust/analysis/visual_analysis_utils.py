import matplotlib.pyplot as plt
import plot_likert
from matplotlib.patches import Circle
from scipy.interpolate import griddata
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import os
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from scipy.stats import linregress, norm, ttest_ind, fisher_exact, binomtest, gamma, sem, mannwhitneyu
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import savgol_filter
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from matplotlib.patches import Patch
from matplotlib.cm import get_cmap
from typing import Union, Sequence, Optional, Dict, Tuple, List
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Wedge, FancyArrowPatch

# Set the default font family
plt.rcParams['font.family'] = 'serif'  # or 'sans-serif', 'monospace', 'cursive', 'fantasy'
plt.rcParams['font.serif'] = ['DejaVu Serif'] # 'Palatino', 'DejaVu Serif', Times New Roman # List of fonts to try
plt.rcParams['font.size'] = 15

def assign_interaction_no(group):
    high_first_low_last = [2,4,6,8,10,12,14,16,20,22,27,29,31,33,35,37,39,41,43,45]
    low_first_high_last = [1,3,7,9,11,13,15,17,19,21,24,26,28,30,32,34,36,38,40,42,44,46]

    pid = group['PID'].iloc[0]
    is_high_first = pid in high_first_low_last

    rows_to_keep = []
    interaction_no = []

    for _, row in group.iterrows():
        if pd.isna(row['AGVname']):
            # skip this row entirely
            continue

        # Convert AGVname to int
        try:
            agv = int(float(row['AGVname']))
        except ValueError:
            agv = int(str(row['AGVname']).split('_')[-1])

        drate = str(row['DRate']).strip()

        if (is_high_first and drate == 'High') or ((not is_high_first) and drate == 'Low'):
            interaction_no.append(agv)
        else:
            interaction_no.append(agv + 16)

        rows_to_keep.append(row)

    # Create a new DataFrame from only the kept rows
    if rows_to_keep:
        result_df = pd.DataFrame(rows_to_keep).copy()
        result_df['Interaction_No'] = interaction_no
        return result_df
    else:
        return pd.DataFrame(columns=list(group.columns) + ['Interaction_No'])


def compute_trust_patterns(data: pd.DataFrame):
    result = []

    for pid, group in data.groupby('PID'):
        group = group.sort_values('StartTime').reset_index(drop=True)
        if len(group) < 2:
            continue  # Skip if not enough points

        trust_vals = group['Trust'].to_numpy()
        trust_series = pd.Series(trust_vals)
        interaction = np.arange(1, len(trust_vals) + 1)

        slope, intercept, r_value, p_value, std_err = linregress(interaction, trust_vals)
        mean_trust = trust_vals.mean()
        std_trust = trust_vals.std()
        range_trust = trust_vals.max() - trust_vals.min()
        mode_trust = trust_series.mode().iloc[0] if not trust_series.mode().empty else np.nan
        median_trust = trust_vals.median() if isinstance(trust_vals, pd.Series) else np.median(trust_vals)
        skewness_trust = trust_series.skew()


        result.append({
            'PID': pid,
            'Mean_Trust': mean_trust,
            'Trust_Slope': slope,
            'Trust_STD': std_trust,
            'Trust_Range': range_trust,
            'Trust_Mode': mode_trust,
            'Trust_Median': median_trust,
            'Trust_Skewness' : skewness_trust })

    return pd.DataFrame(result)

def classify_trust_patterns(df_summary: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
    # Select relevant features for clustering
    features = ['Mean_Trust', 'Trust_Slope', 'Trust_STD', 'Trust_Range', 'Trust_Mode', 'Trust_Median', 'Trust_Skewness']
    X = df_summary[features].copy()

    # Handle missing values if any
    X = X.fillna(X.mean())

    # Normalize features + fit Gaussian Mixture model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GaussianMixture(n_components=n_clusters, random_state=42)
    model.fit(X_scaled)
    cluster_labels = model.predict(X_scaled)

    # Add cluster labels as Trust_Pattern
    df_summary['Trust_Pattern'] = cluster_labels

    return df_summary

def get_cluster_label(cluster_id, cluster_file: pd.DataFrame):
    """
    Return behavioral cluster label with count.
    """
    count = (cluster_file['Cluster'] == cluster_id).sum()

    label_map = {
        1: "Skeptical",
        2: "Deliberate"
    }

    label = label_map.get(cluster_id, f"Cluster {cluster_id}")
    return f"{label} (N={count})"

def plot_combined_mean_DVs_with_clusters(
    df: pd.DataFrame,
    cluster_file: pd.DataFrame,
    save_path: str = None,
    fontsize: int = 12
) -> plt.Figure:

    # Load and merge cluster info
    cluster_df = cluster_file[['PID', 'Cluster']]
    df = df.copy()
    df = df.merge(cluster_df, on='PID', how='inner')

    # Assign interaction number per PID
    df = df.sort_values(['PID', 'StartTime']).reset_index(drop=True)
    df['Interaction'] = df.groupby('PID').cumcount() + 1

    features = ['Trust', 'Safe', 'Comfort', 'Expect']
    feature_representitative = ['Trust', 'Safety', 'Comfort', 'Expectancy']

    fig, axs = plt.subplots(2, 2, figsize=(15, 8), sharex=True, sharey=True)
    axs = axs.flatten()

    # For solid mean lines
    mean_line_styles = {
        1: {'color': 'tab:blue', 'linestyle': '-', 'linewidth': 2},
        2: {'color': 'tab:orange', 'linestyle': '--', 'linewidth': 2},
        3: {'color': 'tab:green', 'linestyle': '--', 'linewidth': 2},
        # Add more if needed
    }

    # For dashed slope lines
    slope_line_styles = {
        1: {'color': 'darkblue', 'linestyle': '-.', 'linewidth': 2.5},
        2: {'color': 'tab:red', 'linestyle': ':', 'linewidth': 2.5},
        3: {'color': 'tab:green', 'linestyle': ':', 'linewidth': 2.5},
        # Add more if needed
    }


    for i, feature in enumerate(features):
        ax = axs[i]
        feature_name = feature_representitative[i]

        for cluster_id in sorted(df['Cluster'].unique()):
            group_data = df[df['Cluster'] == cluster_id]

            # Per-interaction stats
            interaction_stats = (
                group_data.groupby('Interaction')[feature]
                .agg(['mean', 'count', 'std'])
                .reset_index()
            )
            interaction_stats['ci95'] = 1.96 * interaction_stats['std'] / np.sqrt(interaction_stats['count'])

            # Plot temporal mean line
            ax.plot(
                interaction_stats['Interaction'],
                interaction_stats['mean'],
                label=get_cluster_label(cluster_id, cluster_file),
                alpha=0.6,
                **mean_line_styles.get(cluster_id, {})
            )

            # Plot CI band around temporal mean
            ax.fill_between(
                interaction_stats['Interaction'],
                interaction_stats['mean'] - interaction_stats['ci95'],
                interaction_stats['mean'] + interaction_stats['ci95'],
                color=mean_line_styles.get(cluster_id, {}).get('color', 'gray'),
                alpha=0.2
            )

            # Fit regression slope
            X = interaction_stats['Interaction'].values
            y = interaction_stats['mean'].values
            degree = 1
            X_poly = np.vander(X, N=degree + 1, increasing=True)
            model = sm.OLS(y, X_poly).fit()
            x_pred = np.linspace(X.min(), X.max(), 100)
            x_pred_poly = np.vander(x_pred, N=degree + 1, increasing=True)
            y_pred = model.predict(x_pred_poly)

            full_label = get_cluster_label(cluster_id, cluster_file)
            cluster_name = full_label.split(" (")[0]  # removes everything from ' (' onward
            slope_label = f"{cluster_name} Linear Slope"

            # Plot regression slope line
            ax.plot(
            x_pred,
            y_pred,
            label=slope_label,
            **slope_line_styles.get(cluster_id, {})
        )


        #ax.set_title(feature)
        if i > 1:
            ax.set_xlabel("Interaction Number")
        MAP = {
            "trust": "Trust in the AGV (1-10)",
            "comfort": "Comfort with the AGV (1-10)",
            "safety": "Safety with the AGV (1-10)",
            "expectancy": "Eexpectancy of the AGV (1-10)",
        }
        ax.set_ylabel(MAP.get(feature_name.lower(), f"Rate your {feature_name} (1–10)"))
        ax.set_ylim(1, 10.5)
        ax.set_xlim(1, 32)
        ax.grid(True)
        # ax.axvline(x=11, linestyle=':', color='black', linewidth=2)

        if i == 3:
            ax.legend(loc='lower right')

    # plt.suptitle("Trust, Safety, Comfort, and Expectancy by Cluster", fontsize=fontsize + 4)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1) 

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    plt.show()

def plot_combined_mean_features_with_clusters(
    df: pd.DataFrame,
    cluster_file: pd.DataFrame,
    save_path: str = None) -> plt.Figure:

    # Load cluster mapping
    cluster_df = cluster_file[['PID', 'Cluster']]
    
    # Merge with main df
    df = df.copy()
    df = df.merge(cluster_df, on='PID', how='inner')

    # Sort and assign interaction number
    df = df.sort_values(['PID', 'StartTime']).reset_index(drop=True)
    df['Interaction'] = df.groupby('PID').cumcount() + 1
    df['User_Relative_Speed'] = df['User_Relative_Speed']/100
    df['GazeDuration'] = df['GazeDuration']/100 

    features = ['User_Relative_Speed', 'mean_agv_spd', 'GazeDuration']
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    axs = axs.flatten()

    # Plot the first three continuous features
    for i, feature in enumerate(features):
        ax = axs[i]
        for cluster_id in sorted(df['Cluster'].unique()):
            group_data = df[df['Cluster'] == cluster_id]
            interaction_stats = group_data.groupby('Interaction')[feature].agg(['mean', 'count', 'std']).reset_index()
            interaction_stats['ci90'] = interaction_stats.apply(
                lambda row: 1.90 * row['std'] / np.sqrt(row['count']) if row['count'] > 1 else 0, axis=1
            )

            ax.plot(interaction_stats['Interaction'], interaction_stats['mean'], label=get_cluster_label(cluster_id, cluster_file))
            ax.fill_between(
                interaction_stats['Interaction'],
                interaction_stats['mean'] - interaction_stats['ci90'],
                interaction_stats['mean'] + interaction_stats['ci90'],
                alpha=0.5
            )

        ax.set_title(f"Mean {feature} with 90% CI")
        ax.set_xlabel("Interaction Number")
        ax.set_ylabel(feature)
        ax.set_ylim(0, 10.5)
        ax.grid(True)
        ax.legend(loc='lower left')

    # Plot for 'Cross_First' — categorical feature
    ax = axs[3]
    valid_entries = df[~df['Cross_First'].isin(['Null', 'null', 'NA', 'N/A']) & df['Cross_First'].notna()]
    cross_counts = valid_entries.groupby(['Cluster', 'Cross_First']).size().unstack(fill_value=0)
    cross_counts.index = [get_cluster_label(c, cluster_file) for c in cross_counts.index]
    cross_counts[['AGV', 'User']].plot(kind='bar', ax=ax, width=0.6, color=["#3f1fb4", "#1ecc6f"])

    ax.set_title("Who Crossed First by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.grid(True, axis='y')
    ax.legend(title="Crossed First")

    plt.suptitle("Behavioral Features and Cross-First Pattern by Cluster", fontsize=16)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)

    plt.tight_layout()
    plt.show()

def trust_heatmap(data: pd.DataFrame, save_path=None) -> Figure:

    # Sort by StartTime per PID to preserve order
    grouped = data.groupby("PID").apply(lambda g: g.sort_values("StartTime")).reset_index(drop=True)

    # Add an interaction index (1–32)
    grouped['Interaction_Index'] = grouped.groupby('PID').cumcount() + 1

    # Pivot to get a PID x Interaction matrix
    heatmap_data = grouped.pivot(index='PID', columns='Interaction_Index', values='Trust')

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(heatmap_data, cmap='RdYlGn', cbar_kws={'label': 'Trust'}, ax=ax)

    ax.set_title('Trust Ratings per Interaction (1–32) per PID')
    ax.set_xlabel('Interaction Number')
    ax.set_ylabel('PID')
    ax.set_xticks(range(0, 32, 5))
    ax.set_xticklabels(range(1, 33, 5))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def apply_smoothing(summary_df, columns, method='savgol', x_col='Time_Index', loess_frac=0.1):
    for col in columns:
        y = summary_df[col].fillna(method="ffill").fillna(method="bfill").to_numpy()

        if method == 'savgol':
            polyorder = 3
            raw_length = max(int(len(y) * 0.15), polyorder + 2)
            window_length = raw_length if raw_length % 2 == 1 else raw_length + 1
            y_smooth = savgol_filter(y, window_length=window_length, polyorder=polyorder, mode="interp")

        elif method == 'loess':
            x = summary_df[x_col].to_numpy()
            y_smooth = lowess(y, x, frac=loess_frac, return_sorted=False)

        else:
            raise ValueError("Unsupported smoothing method. Choose 'savgol' or 'loess'.")

        summary_df[f"{col}_smoothed"] = y_smooth

    return summary_df

def plot_all_cluster_features(
    df: pd.DataFrame,
    cluster_file: pd.DataFrame,
    feature_cols: list,
    smoothing='loess',
    loess_frac=0.1,
    save_path=None,
    ci: float = 1.96,
    alpha: float = 0.14,
    start_index_time=21,
    window_size=31
):
    df = df.merge(cluster_file, on="PID", how="left")

    if "Timestamp" not in df.columns or "Interaction_No" not in df.columns:
        raise ValueError("The dataframe must contain 'Timestamp' and 'Interaction_No' columns.")

    df = df.sort_values(['PID', 'Timestamp']).copy()
    if 'Time_Index' not in df.columns:
        df['Time_Index'] = df.groupby('PID').cumcount() + 1

    # Fixed-position switch lines
    n_lines = 32
    switch_lines = [(start_index_time + i * window_size - 1, i + 1) for i in range(n_lines)]

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), sharex='col', sharey=False)
    axes = axes.flatten()

    for idx, feature in enumerate(feature_cols):
        ax = axes[idx]
        feature_name = feature.replace("_", " ")

        summaries = []
        for clust, group in df.groupby("Cluster", dropna=False):
            # 1) Group by Time_Index
            # 2) Pandas mean/std skip NaN by default (i.e., use only non-NaN values)
            # 3) Compute mean, std, n, then CI
            agg = (
                group.groupby('Time_Index', as_index=False)[feature]
                     .agg(mean='mean', sd='std', n='count')
            )
            # Avoid div-by-zero for n=0
            agg['se'] = agg['sd'] / np.sqrt(agg['n'].clip(lower=1))
            agg['lwr'] = agg['mean'] - ci * agg['se']
            agg['upr'] = agg['mean'] + ci * agg['se']
            agg['Cluster'] = clust

            # Optional smoothing
            agg = apply_smoothing(
                agg,
                columns=['mean', 'lwr', 'upr'],
                method=smoothing,
                x_col='Time_Index',
                loess_frac=loess_frac
            )
            summaries.append(agg)

        if not summaries:
            continue

        summary_df = pd.concat(summaries, ignore_index=True)

        unique_clusters = sorted(summary_df['Cluster'].dropna().unique().tolist())
        for clust, grp in summary_df.groupby("Cluster", dropna=False):
            linestyle = '--' if len(unique_clusters) > 1 and clust == unique_clusters[1] else '-'
            y_mean = grp.get("mean_smoothed", grp["mean"])
            y_lwr  = grp.get("lwr_smoothed",  grp["lwr"])
            y_upr  = grp.get("upr_smoothed",  grp["upr"])

            ax.plot(
                grp["Time_Index"],
                y_mean,
                label=get_cluster_label(clust, cluster_file),
                linewidth=1.5,
                linestyle=linestyle
            )
            ax.fill_between(grp["Time_Index"], y_lwr, y_upr, alpha=alpha)

        # Vertical switch markers
        for x_pos, label in switch_lines:
            ax.axvline(x=x_pos, color='black', linestyle='--', linewidth=1, alpha=0.3)

        ax.set_xlim(0, 992)
        ax.set_xticks(np.arange(0, 993, 100))
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{feature_name} (Normalized)", fontsize=14, fontfamily='serif', labelpad=10)
        ax.tick_params(axis='both', labelsize=12)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily('serif')
        if idx >= 2:
            ax.set_xlabel("Time Relative to AGV Interaction (s)", fontsize=14, fontfamily='serif', labelpad=10)
        if idx == 3:
            ax.legend(fontsize=12, prop={'family': 'serif'})

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_other_features(
    df: pd.DataFrame,
    cluster_file: pd.DataFrame,
    smoothing: str = 'loess',
    loess_frac: float = 0.1,
    save_path: str = None,
    ci: float = 1.96,
    alpha: float = 0.14,
    start_index_time: int = 21,
    window_size: int = 31,
    pid_col: str = "PID",
    cluster_col: str = "Cluster",
    time_col: str = "Time_Index",
    timestamp_col: str = "Timestamp",
    interaction_col: str = "Interaction_No",
    cumulative_col: str = "Cumulative_Gaze_on_AGV",
    eye_target_col: str = "Eye_Target",
    agv_keyword: str = "AGV",
    ttc_col: str = "Time_to_Collision",
):
    """
    1x3 visualization:
      (1) normalized cumulative gaze on AGV (global min-max)
      (2) proportion of gaze on AGV (Eye_Target contains agv_keyword)
      (3) TTC risk = 1 / (1 + TTC)  (inf TTC -> risk 0), aggregated by cluster over time

    Expects df to already contain:
      - cumulative_col (e.g., Cumulative_Gaze_on_AGV)
      - eye_target_col (e.g., Eye_Target)
      - ttc_col (e.g., time_to_collision)
    """

    # ---- merge clusters ----
    df = df.merge(cluster_file[[pid_col, cluster_col]], on=pid_col, how="left")

    # ---- basic checks ----
    if timestamp_col not in df.columns or interaction_col not in df.columns:
        raise ValueError(f"The dataframe must contain '{timestamp_col}' and '{interaction_col}' columns.")

    needed = {cumulative_col, eye_target_col, ttc_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for plotting: {sorted(missing)}")

    # ---- sort & define time index ----
    df = df.sort_values([pid_col, timestamp_col]).copy()
    if time_col not in df.columns:
        df[time_col] = df.groupby(pid_col).cumcount() + 1

    # ---- global min-max normalization for cumulative gaze ----
    vals = df[cumulative_col].astype(float).to_numpy()
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)
    norm_col = f"{cumulative_col}_norm"
    if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
        df[norm_col] = (df[cumulative_col].astype(float) - vmin) / (vmax - vmin)
    else:
        df[norm_col] = 0.0

    # ---- switch lines (your existing logic) ----
    n_lines = 32
    switch_lines = [(start_index_time + i * window_size - 1, i + 1) for i in range(n_lines)]

    # ---- 1x3 figure ----
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5), sharex=True, sharey=False)

    # ---------------------------------------------------------------------
    # Panel 1: normalized cumulative gaze
    # ---------------------------------------------------------------------
    left_parts = []
    for clust, g in df.groupby(cluster_col, dropna=False):
        agg = (g.groupby(time_col, as_index=False)[norm_col]
                 .agg(mean='mean', sd='std', n='count'))
        agg['se']  = agg['sd'] / np.sqrt(agg['n'].clip(lower=1))
        agg['lwr'] = agg['mean'] - ci * agg['se']
        agg['upr'] = agg['mean'] + ci * agg['se']
        agg[cluster_col] = clust
        agg = apply_smoothing(
            agg, ['mean', 'lwr', 'upr'],
            method=smoothing, x_col=time_col, loess_frac=loess_frac
        )
        left_parts.append(agg)

    if left_parts:
        L = pd.concat(left_parts, ignore_index=True)
        uniq = sorted(L[cluster_col].dropna().unique().tolist())
        for clust, grp in L.groupby(cluster_col, dropna=False):
            ls = '--' if len(uniq) > 1 and clust == uniq[1] else '-'
            y_mean = grp.get("mean_smoothed", grp["mean"])
            y_lwr  = grp.get("lwr_smoothed",  grp["lwr"])
            y_upr  = grp.get("upr_smoothed",  grp["upr"])
            ax1.plot(
                grp[time_col], y_mean,
                label=get_cluster_label(clust, cluster_file),
                linewidth=1.5, linestyle=ls
            )
            ax1.fill_between(grp[time_col], y_lwr, y_upr, alpha=alpha)

    for x_pos, _ in switch_lines:
        ax1.axvline(x=x_pos, color='black', linestyle='--', linewidth=1, alpha=0.3)

    ax1.set_xlim(0, 992)
    ax1.set_xticks(np.arange(0, 993, 100))
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Cumulative Gaze on AGV (Global Min–Max)", fontsize=13, fontfamily='serif', labelpad=10)
    ax1.tick_params(axis='both', labelsize=12)
    for t in ax1.get_xticklabels() + ax1.get_yticklabels():
        t.set_fontfamily('serif')

    # ---------------------------------------------------------------------
    # Panel 2: proportion gaze on AGV (Eye_Target contains keyword)
    # ---------------------------------------------------------------------
    on_agv = df[eye_target_col].astype(str).str.contains(agv_keyword, case=False, na=False)
    pid_time = (
        pd.DataFrame({
            pid_col: df[pid_col],
            cluster_col: df[cluster_col],
            time_col: df[time_col],
            "_on_agv": on_agv.astype(int)
        })
        .groupby([pid_col, cluster_col, time_col], as_index=False)["_on_agv"].max()
    )

    right_parts = []
    for clust, g in pid_time.groupby(cluster_col, dropna=False):
        agg = (g.groupby(time_col, as_index=False)["_on_agv"]
                 .agg(mean='mean', sd='std', n='count'))
        agg['sd']  = agg['sd'].fillna(0.0)
        agg['se']  = agg['sd'] / np.sqrt(agg['n'].clip(lower=1))
        agg['lwr'] = (agg['mean'] - ci * agg['se']).clip(0, 1)
        agg['upr'] = (agg['mean'] + ci * agg['se']).clip(0, 1)
        agg[cluster_col] = clust
        agg = apply_smoothing(
            agg, ['mean', 'lwr', 'upr'],
            method=smoothing, x_col=time_col, loess_frac=loess_frac
        )
        right_parts.append(agg)

    if right_parts:
        R = pd.concat(right_parts, ignore_index=True)
        uniq_r = sorted(R[cluster_col].dropna().unique().tolist())
        for clust, grp in R.groupby(cluster_col, dropna=False):
            ls = '--' if len(uniq_r) > 1 and clust == uniq_r[1] else '-'
            y_mean = grp.get("mean_smoothed", grp["mean"])
            y_lwr  = grp.get("lwr_smoothed",  grp["lwr"])
            y_upr  = grp.get("upr_smoothed",  grp["upr"])
            ax2.plot(
                grp[time_col], y_mean,
                label=get_cluster_label(clust, cluster_file),
                linewidth=1.5, linestyle=ls
            )
            ax2.fill_between(grp[time_col], y_lwr, y_upr, alpha=alpha)

    for x_pos, _ in switch_lines:
        ax2.axvline(x=x_pos, color='black', linestyle='--', linewidth=1, alpha=0.3)

    ax2.set_xlim(0, 992)
    ax2.set_xticks(np.arange(0, 993, 100))
    ax2.set_ylim(0, 1)
    ax2.set_ylabel(f"Proportion of Gaze on {agv_keyword}", fontsize=13, fontfamily='serif', labelpad=10)
    ax2.tick_params(axis='both', labelsize=12)
    for t in ax2.get_xticklabels() + ax2.get_yticklabels():
        t.set_fontfamily('serif')

    # ---------------------------------------------------------------------
    # Panel 3: TTC Risk = 1 / (1 + TTC)
    #   - inf TTC -> 0 risk automatically
    #   - NaN TTC -> 0 risk (conservative / avoids dropping rows)
    # ---------------------------------------------------------------------
    ttc_vals = pd.to_numeric(df[ttc_col], errors='coerce').astype(float)
    df["_ttc_risk"] = 1.0 / (1.0 + ttc_vals)
    df["_ttc_risk"] = df["_ttc_risk"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    risk_parts = []
    for clust, g in df.groupby(cluster_col, dropna=False):
        agg = (g.groupby(time_col, as_index=False)["_ttc_risk"]
                 .agg(mean='mean', sd='std', n='count'))
        agg['sd']  = agg['sd'].fillna(0.0)
        agg['se']  = agg['sd'] / np.sqrt(agg['n'].clip(lower=1))
        agg['lwr'] = (agg['mean'] - ci * agg['se']).clip(0, 1)
        agg['upr'] = (agg['mean'] + ci * agg['se']).clip(0, 1)
        agg[cluster_col] = clust
        agg = apply_smoothing(
            agg, ['mean', 'lwr', 'upr'],
            method=smoothing, x_col=time_col, loess_frac=loess_frac
        )
        risk_parts.append(agg)

    if risk_parts:
        RR = pd.concat(risk_parts, ignore_index=True)
        uniq_t = sorted(RR[cluster_col].dropna().unique().tolist())
        for clust, grp in RR.groupby(cluster_col, dropna=False):
            ls = '--' if len(uniq_t) > 1 and clust == uniq_t[1] else '-'
            y_mean = grp.get("mean_smoothed", grp["mean"])
            y_lwr  = grp.get("lwr_smoothed",  grp["lwr"])
            y_upr  = grp.get("upr_smoothed",  grp["upr"])
            ax3.plot(
                grp[time_col], y_mean,
                label=get_cluster_label(clust, cluster_file),
                linewidth=1.5, linestyle=ls
            )
            ax3.fill_between(grp[time_col], y_lwr, y_upr, alpha=alpha)

    for x_pos, _ in switch_lines:
        ax3.axvline(x=x_pos, color='black', linestyle='--', linewidth=1, alpha=0.3)

    ax3.set_xlim(0, 992)
    ax3.set_xticks(np.arange(0, 993, 100))
    ax3.set_ylim(0, 1)
    ax3.set_ylabel(r"Collision Risk ($1/(1+\mathrm{TTC})$)", fontsize=13, fontfamily='serif', labelpad=10)
    ax3.tick_params(axis='both', labelsize=12)
    for t in ax3.get_xticklabels() + ax3.get_yticklabels():
        t.set_fontfamily('serif')

    # ---- common x-label + legend ----
    for ax in (ax1, ax2, ax3):
        ax.set_xlabel("Time Relative to AGV Interaction (s)", fontsize=13, fontfamily='serif', labelpad=10)

    # One legend for the whole figure (avoid duplicates)
    ax3.legend(fontsize=12, prop={'family': 'serif'}, loc='best')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

    # Cleanup temp column
    df.drop(columns=["_ttc_risk"], inplace=True, errors="ignore")

def plot_efficiency_vs_safety(
    data: pd.DataFrame,
    cluster_file: pd.DataFrame,
    distance_file: pd.DataFrame,
    save_path = None,
):
    """
    data: one row per interaction (has PID, Interaction_No, StartTime, EndTime, etc.)
    distance_file: time series rows (PID, Interaction_No, User_X, User_Y, and an order column like Timestamp/Time_Index)
    cluster_file: (PID, Cluster)
    """
    if 'Interaction_No' not in distance_file.columns:
        required_cols = {'PID','AGVname','DRate'}
        missing = required_cols - set(distance_file.columns)
    if missing:
        raise ValueError(f"distance_file is missing columns: {missing}")

    distance_file = (
    distance_file
    .groupby('PID', group_keys=False)
    .apply(assign_interaction_no)
    )


    # (Optional) make it int and ordered
    distance_file['Interaction_No'] = distance_file['Interaction_No'].astype(int)
    # --- 1) Compute distance traveled per PID × Interaction_No ---
    pos = distance_file.copy()

    # Pick an ordering column for consecutive steps
    order_cols = [c for c in ['Timestamp','Time_Index','Frame','t','SampleIndex'] if c in pos.columns]
    if order_cols:
        pos = pos.sort_values(['PID','Interaction_No'] + order_cols)
    else:
        # Fall back to original row order within groups
        pos = pos.sort_values(['PID','Interaction_No']).copy()
        pos['_row_order'] = pos.groupby(['PID','Interaction_No']).cumcount()
        order_cols = ['_row_order']

    # Step distances within each trajectory
    pos['dx'] = pos.groupby(['PID','Interaction_No'])['User_X'].diff()
    pos['dy'] = pos.groupby(['PID','Interaction_No'])['User_Y'].diff()
    pos['step_dist'] = np.sqrt(pos['dx']**2 + pos['dy']**2)

    dist_summary = (
        pos.groupby(['PID','Interaction_No'], as_index=False)['step_dist']
           .sum(min_count=1)  # NaNs-safe sum
           .rename(columns={'step_dist':'Distance_Traveled'})
    )

    def _minmax_per_interaction(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            # if all values equal or NaN, set to 0
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    dist_summary['Distance_Traveled_Norm'] = (
        dist_summary
        .groupby('Interaction_No')['Distance_Traveled']
        .transform(_minmax_per_interaction)
    )

    # --- 3) Merge with interaction-level data + clusters ---
    df = data.copy()
    df = df.merge(cluster_file[['PID','Cluster']], on='PID', how='left')
    df = df.merge(dist_summary, on=['PID','Interaction_No'], how='left')

    # Interaction time (s)
    if 'StartTime' in df.columns and 'EndTime' in df.columns:
        df['Interaction_Time'] = (
            pd.to_datetime(df['EndTime']) - pd.to_datetime(df['StartTime'])
        ).dt.total_seconds()

    blue = "#1f77b4"
    orange = "#ff7f0e"

    cluster_colors = {
        1: blue,
        2: orange,
    }

    # --- Plotting: Interaction time vs normalized distance traveled ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

    interaction_range = sorted(df['Interaction_No'].dropna().unique())
    features = ['Interaction_Time', 'Distance_Traveled_Norm']
    ylabels  = ['Trial Time (s)', 'Distance Traveled (Normalized)']

    cluster_ids = sorted(df['Cluster'].dropna().unique())

    for ax_idx, (ax, feature, ylabel) in enumerate(zip(axes, features, ylabels)):

        for cluster_id in cluster_ids:
            cluster_id_int = int(cluster_id)
            sub = df[df['Cluster'] == cluster_id]

            agg = (
                sub.groupby('Interaction_No')[feature]
                .agg(['mean', 'count', 'std'])
                .reindex(interaction_range)
            )
            agg['sem'] = agg['std'] / np.sqrt(agg['count'].clip(lower=1))

            c = cluster_colors.get(cluster_id_int, "0.4")

            # Cluster 1 solid, Cluster 2 dashed
            ls = '-' if cluster_id_int == 1 else '--'

            # Only add labels on the RIGHT plot (so legend appears once)
            label = get_cluster_label(cluster_id_int, cluster_file) if ax_idx == 1 else None

            ax.errorbar(
                agg.index, agg['mean'], yerr=agg['sem'],
                capsize=4, marker='o', linestyle=ls,
                color=c, ecolor=c,
                label=label
            )

        ax.set_xlabel("Interaction Number")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Legend only on the right subplot
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_categorical_survey_responses(
    df,
    cluster_file,
    save_path=None,
    custom_labels=None  # dict like {"VRExperience": ["Unfamiliar", "Very inexper.", ...]}
):
    
    custom_labels = {
    "GamingFrequency": [
        "Never", "VeryRrarely", "Rarely", "Occasionally",
        "Frequently", "Moderately Frequently",
        "Very Frequently", "I prefer not to answer."
    ],
    "VRHeadsetExperience": [
        "Unfamiliar", "Very Inexperienced", "Inexperienced",
        "Neither",
        "Moderately Experienced", "Experienced", "Very Experienced",
        "I prefer not to answer."
    ],
    "VRExperience": [
        "Unfamiliar", "Very Inexperienced", "Inexperienced",
        "Neither",
        "**", "Experienced", "Very Experienced", # ** Moderately experienced
        "I prefer not to answer."
    ],
    "AGVInteraction": [
        "*", "Very Rarely", "Rarely", "Occasionally", # * Never
        "Frequently", "Moderately Frequently",
        "Very Frequently", "I prefer not to answer."
    ]
    }

    # Merge clusters into df
    df = df.merge(cluster_file[['PID', 'Cluster']], on='PID', how='left')

    # Target columns & pretty names
    cols = [
        'GamingFrequency',
        'AGVInteraction',
        'VRExperience',
        'VRHeadsetExperience'
    ]
    col_representitative = [
        'Gaming Frequency',
        'Experience with AGVs',
        'VR Experience',
        'VR Headset Experience'
    ]

    # RAW category order used for counting (must match your data EXACTLY)
    experience_order = [
        "Unfamiliar", "Very Inexperienced", "Inexperienced",
        "Neither Inexperienced or Experienced",
        "Moderately experienced", "Experienced", "Very Experienced",
        "I prefer not to answer."
    ]
    frequency_order = [
        "Never", "Very rarely", "Rarely", "Occasionally",
        "Frequently", "Moderately frequently",
        "Very frequently", "I prefer not to answer."
    ]

    # Default display labels (can be same as raw, or shorten/beautify)
    default_experience_labels = experience_order[:]  # or customize here if you like
    default_frequency_labels = frequency_order[:]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        ax = axes[i]

        # One row per PID for this column
        user_df = (
            df[['PID', col, 'Cluster']]
            .drop_duplicates(subset='PID')
            .dropna(subset=[col, 'Cluster'])
            .copy()
        )
        user_df['Cluster'] = user_df['Cluster'].astype(int)

        # Choose the raw order for counts and the default display labels
        if col in ['GamingFrequency', 'AGVInteraction']:
            likert_order = frequency_order
            default_labels = default_frequency_labels
        else:
            likert_order = experience_order
            default_labels = default_experience_labels

        # Pick display labels:
        # if user provided per-question labels, use those (and validate length), else default
        labels = custom_labels.get(col, default_labels)
        if len(labels) != len(likert_order):
            # length mismatch → fallback to default to avoid misalignment
            labels = default_labels

        # Count per category & cluster
        count_data = (
            user_df
            .groupby([col, 'Cluster'])
            .size()
            .reset_index(name='Count')
        )

        # Cluster totals for THIS question (denominator for %)
        totals = (
            user_df
            .groupby('Cluster')
            .size()
            .reindex([1, 2], fill_value=0)
            .rename('Total')
            .reset_index()
        )

        # Compute percentages
        count_pct = count_data.merge(totals, on='Cluster', how='left')
        count_pct['Percent'] = np.where(
            count_pct['Total'] > 0,
            (count_pct['Count'] / count_pct['Total']) * 100.0,
            0.0
        )

        # Pivot: rows = fixed raw categories, cols = clusters
        pivot = pd.DataFrame(index=likert_order)
        for cluster_id in [1, 2]:
            c_series = (
                count_pct[count_pct['Cluster'] == cluster_id]
                .set_index(col)['Percent']
            )
            pivot[f'Cluster {cluster_id}'] = pivot.index.map(c_series).fillna(0.0)

        # --- Plotting ---
        bar_width = 0.4
        blue = "#1f77b4"
        orange = "#ff7f0e"

        for j, raw_val in enumerate(likert_order):
            h1 = float(pivot.loc[raw_val, 'Cluster 1']) if 'Cluster 1' in pivot.columns else 0.0
            h2 = float(pivot.loc[raw_val, 'Cluster 2']) if 'Cluster 2' in pivot.columns else 0.0

            # Bars
            ax.bar(j - bar_width/2, h1, width=bar_width, color=blue, edgecolor='black')
            ax.bar(j + bar_width/2, h2, width=bar_width, color=orange, edgecolor='black', hatch='//')

            # Dashed guide and custom label
            max_h = max(h1, h2)
            ax.plot([j, j], [0, max_h + 3], color='gray', linestyle='--', linewidth=1)
            ax.text(j, max_h + 5, labels[j], ha='center', va='bottom', rotation=90)

        # Axis styling
        ax.set_xticks([])
        ax.set_xlabel('')
        ax.set_ylabel(f"{col_representitative[i]} (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis='y', linestyle=':', alpha=0.5)

        # Legend once
        if i == 3:
            legend_elements = [
                Patch(facecolor=blue, edgecolor='black', label=get_cluster_label(1, cluster_file)),
                Patch(facecolor=orange, edgecolor='black', hatch='//', label=get_cluster_label(2, cluster_file))
            ]
            ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2, hspace=0.1)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
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

def plot_spatiotemporal_trajectories(df, save_path=None):
    df = df.copy()

    # Ensure necessary columns exist
    required_columns = {'PID', 'AGVname', 'DRate', 'Timestamp',
                        'User_X', 'User_Y', 'AGV_X', 'AGV_Y'}

    if not required_columns.issubset(df.columns):
        raise ValueError(f"Missing one of the required columns: {required_columns}")

    # Drop rows with missing coordinate or timestamp data
    df_clean = df.dropna(subset=['User_X', 'User_Y', 'AGV_X', 'AGV_Y', 'Timestamp'])

    # Identify unique group combinations
    group_keys = df_clean.groupby(['PID', 'AGVname', 'DRate']).size().reset_index().sample(frac=1, random_state=42)

    # Choose 4 Low and 4 High DRate groups
    low_drate_groups = group_keys[group_keys['DRate'] == 'Low'].sample(n=4)
    high_drate_groups = group_keys[group_keys['DRate'] == 'High'].sample(n=4)
    selected_groups = pd.concat([low_drate_groups, high_drate_groups])

    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, (_, row) in enumerate(selected_groups.iterrows()):
        pid, agv, drate = row['PID'], row['AGVname'], row['DRate']
        group_df = df_clean[(df_clean['PID'] == pid) &
                            (df_clean['AGVname'] == agv) &
                            (df_clean['DRate'] == drate)].sort_values(by='Timestamp')

        # Normalize color by rank order in time
        norm = plt.Normalize(0, len(group_df) - 1)
        colors1 = plt.cm.Greens(norm(np.arange(len(group_df))))
        colors2 = plt.cm.Blues(norm(np.arange(len(group_df))))


        ax = axes[i]

        # Define custom legend colors
        legend_elements = [
            Patch(facecolor='green', label='AGV Data'),
            Patch(facecolor='blue', label='User Actual Trajectory')
        ]

        # Your scatter plots (colors1 and colors2 can still be gradients or colormaps)
        scatter1 = ax.scatter(
            group_df['AGV_X'] / 100, group_df['AGV_Y'] / 100, 
            c=colors1, marker='.', alpha=0.6
        )

        scatter2 = ax.scatter(
            group_df['User_X'] / 100, group_df['User_Y'] / 100, 
            c=colors2, marker='.', alpha=0.6
        )
    
        # Expected User Trajectory
        if len(group_df) > 0:
            User_X0 = group_df['User_X'].iloc[10] / 100
            User_Y0 = group_df['User_Y'].iloc[10] / 100
            User_Xn = group_df['User_X'].iloc[-1] / 100
            User_Yn = group_df['User_Y'].iloc[-1] / 100

            ax.annotate(
                '', xy=(User_Xn, User_Yn), xytext=(User_X0, User_Y0),
                arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->',
                                linewidth=1.7, linestyle='--'),
                label='User Expected Trajectory'
            )

        # AGV Trajectory Direction Arrow
        if len(group_df) > 75:
            AGV_Xn_1 = group_df['AGV_X'].iloc[10] / 100
            AGV_Yn_1 = group_df['AGV_Y'].iloc[10] / 100
            AGV_Xn = group_df['AGV_X'].iloc[250] / 100
            AGV_Yn = group_df['AGV_Y'].iloc[250] / 100

            '''
            ax.annotate(
                '', xy=(AGV_Xn, AGV_Yn), xytext=(AGV_Xn_1, AGV_Yn_1),
                arrowprops=dict(facecolor='black', edgecolor='black',
                                arrowstyle='->', linewidth=2)
            )
            '''

        ax.set_xlabel('X Coordinate (m)', fontsize=10)
        ax.set_ylabel('Y Coordinate (m)', fontsize=10)
        ax.set_title(f'AGV {agv}, PID {pid}, DRate {drate}', fontsize=12)
        ax.grid(True)
        ax.legend(handles=legend_elements, fontsize=10)
        ax.set_xlim(0, 175)
        ax.set_ylim(0, 150)
        ax.set_xticks(np.arange(0, 180, 20))
        ax.set_yticks(np.arange(0, 160, 20))

    plt.subplots_adjust(left=0.1, right=0.88, top=0.9, bottom=0.1, hspace=0.4)

    if save_path:
        if os.path.isdir(save_path):
            filename = "spatiotemporal_trajectories.png"
            save_path = os.path.join(save_path, filename)
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")

    plt.show()


def p_to_stars(p: float) -> str:
    if p < 0.001:
        # return "***"
        return r"$p < 0.001$"
    if p < 0.01:
        # return "**"
        return r"$p < 0.01$"
    if p < 0.05:
        # return "*"
        return r"$p < 0.05$"
    return ""

def plot_cross_first_by_cluster(
    df,
    cluster_file,
    id_col: str = "PID",
    cluster_col: str = "Cluster",
    decision_col: str = "Cross_First",
    figsize: Tuple[int, int] = (15, 6),
    title: str = "Right of Way distribution by Cluster",
    save_path=None
) -> Tuple[pd.DataFrame, List[str]]:

    df_clusters = cluster_file
    df_interactions = df

    # --- Validate columns
    req1 = {id_col, cluster_col}
    req2 = {id_col, decision_col}
    miss1 = req1 - set(df_clusters.columns)
    miss2 = req2 - set(df_interactions.columns)
    if miss1:
        raise ValueError(f"Cluster assignments missing columns: {miss1}")
    if miss2:
        raise ValueError(f"Interaction data missing columns: {miss2}")

    # --- Merge to attach cluster to each interaction
    merged = pd.merge(
        df_interactions[[id_col, decision_col]],
        df_clusters[[id_col, cluster_col]],
        on=id_col,
        how="left"
    ).dropna(subset=[cluster_col])

    # --- Collect unique raw values (before collapsing)
    unique_raw = sorted(merged[decision_col].dropna().astype(str).unique().tolist())

    # --- Collapse anything not {User, AGV} → 'N/A' (case-insensitive)
    def to_collapsed(val):
        if pd.isna(val):
            return "N/A"
        s = str(val).strip().lower()
        if s == "user":
            return "User"
        if s == "agv":
            return "AGV"
        return "N/A"

    merged["Cross_First_collapsed"] = merged[decision_col].map(to_collapsed)

    categories = ["User", "AGV", "N/A"]

    # --- Aggregate counts and percentages per cluster
    counts = (
        merged.groupby([cluster_col, "Cross_First_collapsed"])
        .size()
        .rename("Count")
        .reset_index()
    )

    # Ensure all categories appear for each cluster (fill 0)
    clusters = sorted(counts[cluster_col].astype(str).unique().tolist())
    if len(clusters) != 2:
        raise ValueError(f"Expected exactly 2 clusters, found {len(clusters)}: {clusters}")

    full_index = pd.MultiIndex.from_product(
        [clusters, categories],
        names=[cluster_col, "Cross_First_collapsed"]
    )

    counts = (
        counts.assign(**{cluster_col: counts[cluster_col].astype(str)})
              .set_index([cluster_col, "Cross_First_collapsed"])
              .reindex(full_index, fill_value=0)
              .reset_index()
    )

    counts["Percent"] = (
        counts.groupby(cluster_col)["Count"]
              .transform(lambda x: 100 * x / x.sum() if x.sum() > 0 else 0.0)
    )

    # --- Extract y1/y2 (Cluster 1/2 counts in ["User","AGV","N/A"] order)
    c1, c2 = clusters[0], clusters[1]

    sub1 = counts[counts[cluster_col] == c1].copy()
    sub2 = counts[counts[cluster_col] == c2].copy()

    sub1["Cross_First_collapsed"] = pd.Categorical(sub1["Cross_First_collapsed"], categories=categories, ordered=True)
    sub2["Cross_First_collapsed"] = pd.Categorical(sub2["Cross_First_collapsed"], categories=categories, ordered=True)

    sub1 = sub1.sort_values("Cross_First_collapsed")
    sub2 = sub2.sort_values("Cross_First_collapsed")

    y1 = sub1["Percent"].to_numpy()
    y2 = sub2["Percent"].to_numpy()

    # --- Pairwise within-cluster comparisons (binomial test on A vs B only)
    # Choose which pairs to compare within each cluster:
    pairs = [("User", "AGV"), ("User", "N/A"), ("AGV", "N/A")]
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    def within_cluster_pair_pvals(y_counts: np.ndarray):
        pvals = {}
        for a, b in pairs:
            ia, ib = cat_to_idx[a], cat_to_idx[b]
            ka, kb = int(y_counts[ia]), int(y_counts[ib])
            n = ka + kb
            if n == 0:
                pvals[(a, b)] = np.nan
                continue
            # H0: P(a) = P(b) within the {a,b} subset => P(a)=0.5
            pvals[(a, b)] = binomtest(ka, n=n, p=0.5, alternative="two-sided").pvalue
        return pvals

    pvals_c1 = within_cluster_pair_pvals(y1)
    pvals_c2 = within_cluster_pair_pvals(y2)

    # --- Plot: ONE AXIS, two blocks. Bars within a cluster touch (bar_width=1.0)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # fig.suptitle(title, y=0.98)

    blue = "#1f77b4"
    orange = "#ff7f0e"

    hatches_c1 = ["o", "o-", "ooo"]   # User, AGV, N/A
    hatches_c2 = ["/", "\\", "||"]    # User, AGV, N/A

    ymax = 100
    yticks = np.arange(0, 101, 10)

    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, ymax)
    ax.set_yticks(yticks)

    bar_width = 1  # makes adjacent bars touch when x positions are consecutive integers

    # Cluster 1 positions: 0,1,2 ; Cluster 2 positions: 4,5,6 (gap only between clusters)
    x1 = np.arange(len(categories))              # 0,1,2
    cluster_gap = 2                              
    x2 = x1 + len(categories) + cluster_gap      # 5? actually 0..2 + 3 +2 => 5..7; OK

    bars1 = ax.bar(x1, y1, width=bar_width, color=blue, edgecolor="black", linewidth=1.0)
    for b, h in zip(bars1, hatches_c1):
        b.set_hatch(h)

    bars2 = ax.bar(x2, y2, width=bar_width, color=orange, edgecolor="black", linewidth=1.0)
    for b, h in zip(bars2, hatches_c2):
        b.set_hatch(h)

    # x labels (repeat categories under each block)
    xticks = np.concatenate([x1, x2])
    ax.set_xticks(xticks)
    category_display_labels = {
    "User": "User",
    "AGV": "AGV",
    "N/A": "No Interaction",   # or "No Crossing", "No Conflict", etc.
    }
    xtick_labels = [category_display_labels[c] for c in categories]
    ax.set_xticklabels(xtick_labels + xtick_labels, rotation=0)


    ax.set_xlabel("Right of Way")
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, ymax)
    ax.set_yticks(yticks)
    ax.set_axisbelow(True)  # make grid go behind bars
    ax.yaxis.grid(True, linestyle='--', linewidth=0.8, alpha=0.35)
    ax.xaxis.grid(False)

    # vertical separator between blocks
    sep_x = (x1[-1] + x2[0]) / 2
    ax.axvline(sep_x, color="0.7", linewidth=1.0)

    # cluster headers
    ax.text(x1.mean(), ymax * 0.98, f"Cluster {c1}", ha="center", va="top", fontsize=13)
    ax.text(x2.mean(), ymax * 0.98, f"Cluster {c2}", ha="center", va="top", fontsize=13)

    # --- draw within-cluster significance brackets
    def add_bracket(ax, x_left, x_right, y_base, text, height=2, text_pad=1.5, lw=1.5):
        y_top = y_base + height
        ax.plot([x_left, x_left, x_right, x_right],
                [y_base, y_top, y_top, y_base],
                color="black", linewidth=lw)
        ax.text((x_left + x_right)/2, y_top + text_pad, text,
                ha="center", va="bottom", fontsize=15)

    # Put these two close comparisons lower, and the wide one higher.
    low_pairs  = [("User", "AGV"), ("AGV", "N/A")]
    high_pairs = [("User", "N/A")]

    # Vertical controls
    base_offset_low  = 15     # lift low brackets above bars
    base_offset_high = 65     # lift high bracket above bars (extra headroom)
    stack_step_low   = 0      # keep low brackets on the same tier (or set small value if they collide)
    stack_step_high  = 0

    def annotate_within_cluster(y_perc, x_positions, pvals_dict):
        low_pairs  = [("User", "AGV"), ("AGV", "N/A")]
        high_pairs = [("User", "N/A")]

        cluster_max = float(np.max(y_perc))

        # percent-scale offsets
        low_offset  = 3    # %
        low_step    = 6    # %
        high_offset = 10   # %

        # keep brackets inside the axis
        cap_low  = ymax - 20   # leave room for bracket height + text
        cap_high = ymax - 10

        # adjacent comparisons
        for level, (a, b) in enumerate(low_pairs):
            p = pvals_dict.get((a, b), np.nan)
            if not np.isfinite(p):
                continue
            stars = p_to_stars(p)
            if not stars:
                continue

            ia, ib = cat_to_idx[a], cat_to_idx[b]
            xa, xb = x_positions[ia], x_positions[ib]
            y_pair_max = max(y_perc[ia], y_perc[ib])

            y_base = min(y_pair_max + low_offset + level * low_step, cap_low)
            add_bracket(ax, xa, xb, y_base, stars, height=2, text_pad=1.5)

        # wide comparison
        for a, b in high_pairs:
            p = pvals_dict.get((a, b), np.nan)
            if not np.isfinite(p):
                continue
            stars = p_to_stars(p)
            if not stars:
                continue

            ia, ib = cat_to_idx[a], cat_to_idx[b]
            xa, xb = x_positions[ia], x_positions[ib]

            y_base = min(cluster_max + high_offset, cap_high)
            add_bracket(ax, xa, xb, y_base, stars, height=2, text_pad=1.5)

    annotate_within_cluster(y1, x1, pvals_c1)
    annotate_within_cluster(y2, x2, pvals_c2)

    # Legend only on the right subplot
    ax.legend(
        handles=[
            Patch(
                facecolor=blue,
                edgecolor="black",
                hatch="o",
                label=get_cluster_label(1, cluster_file)
            ),
            Patch(
                facecolor=orange,
                edgecolor="black",
                hatch="/",
                label=get_cluster_label(2, cluster_file)
            ),
        ],
        loc="upper left",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.show()


def plot_cross_first_between_clusters(
    df: pd.DataFrame,
    cluster_file: pd.DataFrame,
    *,
    id_col: str = "PID",
    cluster_col: str = "Cluster",
    decision_col: str = "Cross_First",
    figsize: Tuple[int, int] = (10, 5),
    title: str | None = None,
    save_path: str | None = None,
):
    """
    Option A (between clusters), participant-level:

    1) Merge cluster labels onto interaction rows.
    2) Collapse decisions into {User, AGV, N/A}.
    3) For each PID, compute %User/%AGV/%N/A across that PID's interactions.
    4) For each category, compare cluster distributions with Mann–Whitney U.
    5) Plot grouped bars: [User, AGV, No Interaction], each group has (Cluster1, Cluster2).

    Returns:
      - pid_props: one row per participant with columns [PID, Cluster, User, AGV, N/A]
      - pvals: dict of p-values per category
    """

    # --- validate columns early (no silent failures)
    if id_col not in df.columns:
        raise ValueError(f"{id_col} not found in df. Available: {df.columns.tolist()}")
    if decision_col not in df.columns:
        raise ValueError(f"{decision_col} not found in df. Available: {df.columns.tolist()}")
    if id_col not in cluster_file.columns:
        raise ValueError(f"{id_col} not found in cluster_file. Available: {cluster_file.columns.tolist()}")
    if cluster_col not in cluster_file.columns:
        raise ValueError(f"{cluster_col} not found in cluster_file. Available: {cluster_file.columns.tolist()}")

    # --- merge to attach cluster to each interaction
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

    # --- collapse categories
    def to_collapsed(val) -> str:
        if pd.isna(val):
            return "N/A"
        s = str(val).strip().lower()
        if s == "user":
            return "User"
        if s == "agv":
            return "AGV"
        return "N/A"

    merged["Cross_First_collapsed"] = merged[decision_col].map(to_collapsed)

    categories = ["User", "AGV", "N/A"]
    category_display = {"User": "User", "AGV": "AGV", "N/A": "No Interaction"}

    # --- participant-level counts -> percents (simple + robust)
    pid_counts = (
        merged.groupby([id_col, cluster_col, "Cross_First_collapsed"])
        .size()
        .unstack("Cross_First_collapsed", fill_value=0)
        .reset_index()
    )

    # Ensure all category columns exist
    for cat in categories:
        if cat not in pid_counts.columns:
            pid_counts[cat] = 0

    totals = pid_counts[categories].sum(axis=1).replace(0, np.nan)
    pid_props = pid_counts[[id_col, cluster_col]].copy()
    for cat in categories:
        pid_props[cat] = (pid_counts[cat] / totals) * 100.0
    pid_props[categories] = pid_props[categories].fillna(0.0)

    # --- cluster list (expect exactly 2)
    clusters = sorted(pid_props[cluster_col].unique().tolist())
    if len(clusters) != 2:
        raise ValueError(f"Expected exactly 2 clusters, found {len(clusters)}: {clusters}")
    c1, c2 = clusters[0], clusters[1]

    # --- between-cluster tests (Option A)
    pvals: Dict[str, float] = {}
    for cat in categories:
        a = pid_props.loc[pid_props[cluster_col] == c1, cat].to_numpy(float)
        b = pid_props.loc[pid_props[cluster_col] == c2, cat].to_numpy(float)
        if len(a) == 0 or len(b) == 0:
            pvals[cat] = np.nan
        else:
            pvals[cat] = mannwhitneyu(a, b, alternative="two-sided").pvalue

    # --- plot (keep styling + hatches)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    blue = "#1f77b4"
    orange = "#ff7f0e"

    hatches_c1 = ["o", "o-", "ooo"]   # User, AGV, N/A
    hatches_c2 = ["/", "\\", "||"]    # User, AGV, N/A

    means_c1 = [pid_props.loc[pid_props[cluster_col] == c1, cat].mean() for cat in categories]
    means_c2 = [pid_props.loc[pid_props[cluster_col] == c2, cat].mean() for cat in categories]

    x = np.arange(len(categories))
    bar_w = 0.42
    x_c1 = x - bar_w / 2
    x_c2 = x + bar_w / 2

    bars1 = ax.bar(x_c1, means_c1, width=bar_w, color=blue, edgecolor="black", linewidth=1.0)
    for b, h in zip(bars1, hatches_c1):
        b.set_hatch(h)

    bars2 = ax.bar(x_c2, means_c2, width=bar_w, color=orange, edgecolor="black", linewidth=1.0)
    for b, h in zip(bars2, hatches_c2):
        b.set_hatch(h)

    ax.set_title(title)
    ax.set_xlabel("Cross First")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([category_display[c] for c in categories], rotation=0)

    ymax = 100
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.35)
    ax.xaxis.grid(False)

    # --- significance brackets per category (cluster1 vs cluster2)
    def add_bracket(ax, x_left, x_right, y_base, text, height=2, text_pad=1.2, lw=1.5):
        y_top = y_base + height
        ax.plot([x_left, x_left, x_right, x_right],
                [y_base, y_top, y_top, y_base],
                color="black", linewidth=lw)
        ax.text((x_left + x_right) / 2, y_top + text_pad, text,
                ha="center", va="bottom", fontsize=14)

    for i, cat in enumerate(categories):
        p = pvals.get(cat, np.nan)
        if not np.isfinite(p):
            continue
        stars = p_to_stars(p)
        if not stars:
            continue
        y_here = max(means_c1[i], means_c2[i])
        y_base = min(y_here + 3, ymax - 8)
        add_bracket(ax, x_c1[i], x_c2[i], y_base, stars)

    # --- legend
    ax.legend(
        handles=[
            Patch(facecolor=blue, edgecolor="black", hatch="o", label=get_cluster_label(int(c1), cluster_file)),
            Patch(facecolor=orange, edgecolor="black", hatch="/", label=get_cluster_label(int(c2), cluster_file)),
        ],
        loc="upper left",
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.show()

    return pid_props, pvals  


def _angle_amount_deg_from_vectors(df, gaze_cols=('GazeDirection_X','GazeDirection_Y'),
                                   user_cols=('GazeOrigin_X','GazeOrigin_Y'),
                                   agv_cols=('AGV_X','AGV_Y')):
    """
    Returns a 1D np.array of unsigned angles in degrees (0..180), one per row of df,
    computed exactly like your snippet: acos( dot / (||gaze|| * ||user->agv||) ).
    If *_Z columns are missing, uses 0 for Z.
    """
    # pull columns safely; if Z missing, create zeros
    def _col(df, name): 
        return df[name].to_numpy() if name in df.columns else np.zeros(len(df), dtype=float)

    gx = df[gaze_cols[0]].to_numpy()
    gy = df[gaze_cols[1]].to_numpy()
    # gz = _col(df, gaze_cols[2])

    ox = df[user_cols[0]].to_numpy()
    oy = df[user_cols[1]].to_numpy()
    # oz = _col(df, user_cols[2])

    ax = df[agv_cols[0]].to_numpy()
    ay = df[agv_cols[1]].to_numpy()
    # az = _col(df, agv_cols[2])

    # vectors
    vx = ax - ox
    vy = ay - oy
    # vz = az - oz

    # dot and norms
    dot = gx*vx + gy*vy #+ gz*vz
    gaze_norm   = np.sqrt(gx*gx + gy*gy) # + gz*gz)
    target_norm = np.sqrt(vx*vx + vy*vy) # + vz*vz)

    cos_theta = dot / (gaze_norm * target_norm + 1e-12)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))  # 0..180 (unsigned)

def animate_user_interactions_with_fov(
    df: pd.DataFrame,
    pid,
    save_path: str = None,
    fps: int = 20,
    divide_by: float = 100.0,        # scale coords (e.g., cm -> m)
    figsize=(14, 6),
    xlim=None,
    ylim=None,
    fov_half_angle_deg: float = 60.0,
    gaze_arrow_len_units: float = 50.0,  # in post-scaled units (e.g., meters)
    angle_col: str = 'Gaze_Angle_to_AGV',
    agv_in_fov_col: str = 'AGV_in_FOV',
    user_cols=('GazeOrigin_X', 'GazeOrigin_Y'),
    agv_cols=('AGV_X', 'AGV_Y'),
    gaze_cols=('GazeDirection_X', 'GazeDirection_Y'),
    show_legend: bool = True,
    fallback_expected: bool = True,   # straight-line expected path if Expected_* has NaNs
):
    """
    Animate (left) trajectories and (right) the FOV view for the same PID.
    Resets when Interaction_No changes, and both subplots share identical x/y limits.

    Required columns:
      ['PID','Interaction_No','Timestamp',
       'User_X','User_Y','AGV_X','AGV_Y',
       'Expected_User_X','Expected_User_Y',
       user_cols..., agv_cols..., gaze_cols..., angle_col, agv_in_fov_col]
    """
    # --- 1) Validate inputs ---
    base_required = {
        'PID','Interaction_No','Timestamp',
        'User_X','User_Y','AGV_X','AGV_Y'
    }
    if not base_required.issubset(df.columns):
        missing = base_required.difference(df.columns)
        raise ValueError(f"Missing required base columns: {missing}")

    # expected may be missing -> we can fallback if requested
    has_expected = {'Expected_User_X','Expected_User_Y'}.issubset(df.columns)
    if not has_expected and not fallback_expected:
        raise ValueError("Expected path columns are missing and fallback_expected=False.")

    # FOV columns required on the right subplot
    fov_required = {angle_col, agv_in_fov_col, user_cols[0], user_cols[1],
                    agv_cols[0], agv_cols[1], gaze_cols[0], gaze_cols[1]}
    missing_fov = fov_required.difference(df.columns)
    if missing_fov:
        raise ValueError(f"Missing required FOV columns: {missing_fov}")

    # --- 2) Filter + scale + sort ---
    dfp = (
        df.loc[df['PID'] == pid]
          .dropna(subset=['User_X','User_Y','AGV_X','AGV_Y','Timestamp'])
          .sort_values(['Interaction_No','Timestamp'])
          .copy()
    )
    if dfp.empty:
        raise ValueError(f"No rows found for PID={pid} after cleaning.")

    # Collect all columns that need dividing, but ensure uniqueness
    to_scale = {'User_X', 'User_Y', 'AGV_X', 'AGV_Y'}
    if {'Expected_User_X','Expected_User_Y'}.issubset(df.columns):
        to_scale |= {'Expected_User_X','Expected_User_Y'}
    to_scale |= {user_cols[0], user_cols[1], agv_cols[0], agv_cols[1]}

    # Coerce to numeric (protect against stray strings) and scale ONCE
    for c in to_scale:
        if c in dfp.columns:
            dfp[c] = pd.to_numeric(dfp[c], errors='coerce') / divide_by

    # --- 3) Split into interactions and build frames ---
    interactions = dfp['Interaction_No'].unique().tolist()
    groups = [dfp[dfp['Interaction_No'] == k] for k in interactions]
    frames = []
    for i, g in enumerate(groups):
        n = len(g)
        frames.extend([(i, t) for t in range(1, n + 1)])

    # --- 4) Axis limits (shared between subplots) ---
    if xlim is None or ylim is None:
        xs = [dfp['User_X'].values, dfp['AGV_X'].values]
        ys = [dfp['User_Y'].values, dfp['AGV_Y'].values]
        if has_expected:
            xs.append(dfp['Expected_User_X'].values)
            ys.append(dfp['Expected_User_Y'].values)
        xs = np.concatenate(xs)
        ys = np.concatenate(ys)
        pad_x = 0.05 * (np.nanmax(xs) - np.nanmin(xs) + 1e-9)
        pad_y = 0.05 * (np.nanmax(ys) - np.nanmin(ys) + 1e-9)
        xlim = (np.nanmin(xs) - pad_x, np.nanmax(xs) + pad_x) if xlim is None else xlim
        ylim = (np.nanmin(ys) - pad_y, np.nanmax(ys) + 1e-9 + pad_y) if ylim is None else ylim

    # --- 5) Figure & axes ---
    fig, (axL, axR) = plt.subplots(1, 2, figsize=figsize)
    suptitle = fig.suptitle("", fontsize=16)

    # ---------- Left subplot (trajectories) ----------
    user_line,     = axL.plot([], [], lw=3, color='blue', label='User (actual)')
    expected_line, = axL.plot([], [], lw=3, color='red', ls='--', label='User (expected)')
    agv_line,      = axL.plot([], [], lw=3, color='green', label='AGV')
    user_tip,      = axL.plot([], [], color='blue', marker='o', ms=10)   # color auto
    expected_tip,  = axL.plot([], [], color='blue', marker='o', ms=10)
    agv_tip,       = axL.plot([], [], color='red', marker='^', ms=10)

    axL.set_xlim(*xlim)
    axL.set_ylim(*ylim)
    axL.set_xlabel('X (m)')
    axL.set_ylabel('Y (m)')
    axL.set_aspect('equal', adjustable='box')
    axL.grid(True)

    # Pre-pack arrays for left subplot
    left_arrays = []
    for g in groups:
        arr = {
            'user_x': g['User_X'].to_numpy(),
            'user_y': g['User_Y'].to_numpy(),
            'agv_x':  g['AGV_X'].to_numpy(),
            'agv_y':  g['AGV_Y'].to_numpy(),
            'label':  g['Interaction_No'].iloc[0]
        }
        if has_expected:
            arr['exp_x'] = g['Expected_User_X'].to_numpy()
            arr['exp_y'] = g['Expected_User_Y'].to_numpy()
        else:
            arr['exp_x'] = np.full_like(arr['user_x'], np.nan, dtype=float)
            arr['exp_y'] = np.full_like(arr['user_y'], np.nan, dtype=float)
        left_arrays.append(arr)

    # ---------- Right subplot (FOV) ----------
    def _wrap360(a):
        return a % 360.0

    def _to_deg_if_needed(a):
        """Heuristic: convert to degrees if value looks like radians."""
        a = float(a)
        return np.degrees(a) if abs(a) <= np.pi + 0.2 else a

    def _pick_arc(gaze_deg, agv_deg, target_angle_deg, tol=2.0):
        """
        Choose the arc from gaze->agv (either CCW or CW) whose length best matches
        abs(target_angle_deg). Returns (theta1, theta2, arc_len_deg).

        - gaze_deg, agv_deg in [0, 360)
        - target_angle_deg can be any float; we'll use its absolute value as the arc length to match.
        - theta1/theta2 are wedge angles in degrees (Matplotlib expects degrees).
        """
        gaze_deg = _wrap360(gaze_deg)
        agv_deg  = _wrap360(agv_deg)
        amount   = abs(float(target_angle_deg))

        # CCW length from gaze to agv
        ccw_len = (agv_deg - gaze_deg) % 360.0
        # CW length is the other way around
        cw_len  = (gaze_deg - agv_deg) % 360.0  # == 360 - ccw_len

        # Pick whichever is closer to the requested amount
        diff_ccw = abs(ccw_len - amount)
        diff_cw  = abs(cw_len  - amount)

        if diff_ccw <= diff_cw:
            # CCW arc
            theta1 = gaze_deg
            theta2 = _wrap360(gaze_deg + ccw_len)
            return theta1, theta2, ccw_len
        else:
            # CW arc (drawn by going from theta1 to theta2 with theta1 <-> theta2 order for Wedge)
            # For CW, we set theta1 = gaze - cw_len, theta2 = gaze
            theta1 = _wrap360(gaze_deg - cw_len)
            theta2 = gaze_deg
            return theta1, theta2, cw_len

    axR.set_xlim(*xlim)
    axR.set_ylim(*ylim)
    axR.set_xlabel('X (m)')
    axR.set_ylabel('Y (m)')
    axR.set_aspect('equal', adjustable='box')
    axR.grid(True)

    # Static legend handles via actual artists we’ll update
    user_scatter = axR.scatter([], [], color='blue', s=100, marker='o', label='User Gaze Origin')
    agv_scatter  = axR.scatter([], [], color='red',  s=100, marker='^', label='AGV')

    # Gaze arrow (green)
    gaze_arrow = FancyArrowPatch((0,0), (0,0), color='green', linewidth=2, mutation_scale=10)
    axR.add_artist(gaze_arrow)

    # FOV boundary rays (blue dashed)
    fov_left  = FancyArrowPatch((0,0), (0,0), linestyle='dashed', color='blue', alpha=0.6, mutation_scale=10)
    fov_right = FancyArrowPatch((0,0), (0,0), linestyle='dashed', color='blue', alpha=0.6, mutation_scale=10)
    axR.add_artist(fov_left)
    axR.add_artist(fov_right)

    # FOV wedge (green translucent)
    fov_wedge = Wedge(center=(0,0), r=gaze_arrow_len_units, theta1=0, theta2=0,
                      facecolor='green', alpha=0.2)
    axR.add_patch(fov_wedge)

    # AGV direction from user (red dashed line)
    agv_dir_line, = axR.plot([], [], color='red', linestyle='--', label='AGV Direction')

    # Angle arc (red translucent) + text
    angle_arc = Wedge(center=(0,0), r=gaze_arrow_len_units * 0.6, theta1=0, theta2=0,
                      edgecolor='red', facecolor='red', alpha=0.2, linestyle='--', linewidth=2)
    axR.add_patch(angle_arc)
    angle_text = axR.text(0, 0, '', color='red', ha='center', va='center')

    if show_legend:
        axL.legend(loc='best')
        axR.legend(loc='upper right')

    # --- Right subplot arrays (ensure numeric angle) ---
    # --- Right subplot arrays (recompute angle amount per row) ---
    right_arrays = []
    for g in groups:
        # IMPORTANT: angle is based on current (already scaled) positions — fine because
        # scaling cancels in the normalization. (If you prefer, compute before scaling.)
        angle_amount_deg = _angle_amount_deg_from_vectors(
            g,
            gaze_cols=('GazeDirection_X','GazeDirection_Y','GazeDirection_Z'),
            user_cols=('GazeOrigin_X','GazeOrigin_Y','GazeOrigin_Z'),
            agv_cols=('AGV_X','AGV_Y','AGV_Z')
        )

        arr = {
            'pid':      g['PID'].iloc[0],
            'label':    g['Interaction_No'].iloc[0],
            'orig_x':   g[user_cols[0]].to_numpy(),
            'orig_y':   g[user_cols[1]].to_numpy(),
            'agv_x':    g[agv_cols[0]].to_numpy(),
            'agv_y':    g[agv_cols[1]].to_numpy(),
            'gaze_x':   g[gaze_cols[0]].to_numpy(),
            'gaze_y':   g[gaze_cols[1]].to_numpy(),
            # use the freshly computed angle AMOUNT (degrees) for drawing the arc
            'angle_amt_deg': angle_amount_deg
        }
        right_arrays.append(arr)

    half_rad = np.radians(fov_half_angle_deg)

    def _safe_norm(vx, vy):
        n = np.hypot(vx, vy)
        if not np.isfinite(n) or n < 1e-9:
            return 1.0  # avoid 0-div; will treat dir as (1,0)
        return n

    def init():
        # Left
        user_line.set_data([], [])
        expected_line.set_data([], [])
        agv_line.set_data([], [])
        user_tip.set_data([], [])
        expected_tip.set_data([], [])
        agv_tip.set_data([], [])
        axL.set_title(f'Interaction')

        # Right
        user_scatter.set_offsets(np.empty((0, 2)))
        agv_scatter.set_offsets(np.empty((0, 2)))
        gaze_arrow.set_positions((0,0), (0,0))
        fov_left.set_positions((0,0), (0,0))
        fov_right.set_positions((0,0), (0,0))
        fov_wedge.set_center((0,0))
        fov_wedge.set_theta1(0); fov_wedge.set_theta2(0)
        agv_dir_line.set_data([], [])
        angle_arc.set_center((0,0)); angle_arc.set_theta1(0); angle_arc.set_theta2(0)
        angle_text.set_text(''); angle_text.set_position((0,0))
        axR.set_title(f'Field of View')

        # Return all animated artists (blit=False below, but still good practice)
        return (
            user_line, expected_line, agv_line, user_tip, expected_tip, agv_tip,
            gaze_arrow, fov_left, fov_right, fov_wedge, agv_dir_line, angle_arc, angle_text
        )

    def update(frame):
        inter_idx, upto = frame
        L = left_arrays[inter_idx]
        R = right_arrays[inter_idx]

        # ---------- Left ----------
        ux, uy = L['user_x'][:upto], L['user_y'][:upto]
        axx, ayy = L['agv_x'][:upto],  L['agv_y'][:upto]
        ex, ey = L['exp_x'][:upto], L['exp_y'][:upto]

        # fallback expected straight line (first->last user) if needed
        if fallback_expected and (not has_expected or np.all(np.isnan(ex)) or np.all(np.isnan(ey))):
            if len(ux) >= 2:
                x0, y0 = ux[0], uy[0]
                x1, y1 = ux[-1], uy[-1]
                t = np.linspace(0, 1, len(ux))
                ex = x0 + (x1 - x0) * t
                ey = y0 + (y1 - y0) * t
            else:
                ex = np.array([]); ey = np.array([])

        user_line.set_data(ux, uy)
        expected_line.set_data(ex, ey)
        agv_line.set_data(axx, ayy)

        if len(ux) > 0:
            user_tip.set_data([ux[-1]], [uy[-1]])
        else:
            user_tip.set_data([], [])
        if len(ex) > 0:
            expected_tip.set_data([ex[-1]], [ey[-1]])
        else:
            expected_tip.set_data([], [])
        if len(axx) > 0:
            agv_tip.set_data([axx[-1]], [ayy[-1]])
        else:
            agv_tip.set_data([], [])

        # axL.set_title(f'PID {pid} — Interaction {L["label"]}')

        # ---------- Right (FOV) ----------
        # Current row
        i = upto - 1
        ox, oy = float(R['orig_x'][i]), float(R['orig_y'][i])
        gx, gy = float(R['gaze_x'][i]), float(R['gaze_y'][i])
        axg, ayg = float(R['agv_x'][i]), float(R['agv_y'][i])

        # Update user/AGV points
        user_scatter.set_offsets(np.array([[ox, oy]]))
        agv_scatter.set_offsets(np.array([[axg, ayg]]))

        # Gaze unit vector
        n = _safe_norm(gx, gy)
        gxn, gyn = (gx / n if n != 0 else 1.0), (gy / n if n != 0 else 0.0)
        gaze_end = (ox + gxn * gaze_arrow_len_units, oy + gyn * gaze_arrow_len_units)
        gaze_arrow.set_positions((ox, oy), gaze_end)

        # FOV bounds
        gaze_angle = np.arctan2(gyn, gxn)
        left_rad  = gaze_angle + np.radians(fov_half_angle_deg)
        right_rad = gaze_angle - np.radians(fov_half_angle_deg)
        left_vec  = (np.cos(left_rad) * gaze_arrow_len_units,  np.sin(left_rad) * gaze_arrow_len_units)
        right_vec = (np.cos(right_rad) * gaze_arrow_len_units, np.sin(right_rad) * gaze_arrow_len_units)
        fov_left.set_positions((ox, oy), (ox + left_vec[0],  oy + left_vec[1]))
        fov_right.set_positions((ox, oy), (ox + right_vec[0], oy + right_vec[1]))

        # FOV wedge
        fov_wedge.set_center((ox, oy))
        fov_wedge.set_theta1(np.degrees(right_rad))
        fov_wedge.set_theta2(np.degrees(left_rad))

        # AGV direction dashed line from user
        vec_u2a = np.array([axg - ox, ayg - oy], dtype=float)
        vn = _safe_norm(vec_u2a[0], vec_u2a[1])
        vxn, vyn = (vec_u2a / vn) if vn != 0 else np.array([1.0, 0.0])
        agv_line_end = (ox + vxn * gaze_arrow_len_units, oy + vyn * gaze_arrow_len_units)
        agv_dir_line.set_data([ox, agv_line_end[0]], [oy, agv_line_end[1]])

        # Angle arc & label
        # ---- Arc that matches the stored angle amount exactly (like plot_agv_user_fov) ----
        # ---- Arc that matches the recomputed angle amount exactly ----
        target_angle_deg = R['angle_amt_deg'][i] if i < len(R['angle_amt_deg']) else np.nan

        if np.isfinite(target_angle_deg):
            # Current headings in degrees from the already-drawn vectors
            gaze_deg = (np.degrees(np.arctan2(gyn, gxn))) % 360.0
            agv_dir  = np.arctan2(vyn, vxn)
            agv_deg  = (np.degrees(agv_dir)) % 360.0

            # Choose CCW or CW arc whose length best matches the angle amount
            theta1, theta2, arc_len = _pick_arc(gaze_deg, agv_deg, target_angle_deg, tol=2.0)

            # Draw arc
            angle_arc.set_center((ox, oy))
            angle_arc.set_theta1(theta1)
            angle_arc.set_theta2(theta2)

            # Label at mid-arc (show the amount you just computed)
            mid_theta = (theta1 + (theta2 - theta1) % 360.0 / 2.0) % 360.0
            r_label = gaze_arrow_len_units * 0.6 + 5.0
            label_x = ox + r_label * np.cos(np.radians(mid_theta))
            label_y = oy + r_label * np.sin(np.radians(mid_theta))
            angle_text.set_position((label_x, label_y))
            angle_text.set_text(f"{target_angle_deg:.1f}°")
        else:
            angle_arc.set_center((ox, oy))
            angle_arc.set_theta1(0.0); angle_arc.set_theta2(0.0)
            angle_text.set_text(''); angle_text.set_position((ox, oy))

        # axR.set_title(f'Cluster 2 Representative — Interaction {R["label"]} — FOV')

        suptitle.set_text(f"Cluster 1 Representative — Interaction {L['label']}")
        return (
            user_line, expected_line, agv_line, user_tip, expected_tip, agv_tip,
            gaze_arrow, fov_left, fov_right, fov_wedge, agv_dir_line, angle_arc, angle_text
        )
    
    # fig.suptitle(f"Cluster 2 Representative — Trajectories & FOV", fontsize=16, y=0.98)

    # Note: blitting with many patches can be fussy; keep it simple:
    anim = FuncAnimation(fig, update, frames=frames, init_func=init, blit=False, interval=int(1000 / fps))

    if save_path:
        ext = os.path.splitext(save_path)[1].lower()
        if ext == '.mp4':
            writer = FFMpegWriter(fps=fps, metadata={'artist': 'Mobina / HIALab'})
            anim.save(save_path, writer=writer)
            plt.close(fig)
        elif ext == '.gif':
            writer = PillowWriter(fps=fps)
            anim.save(save_path, writer=writer)
            plt.close(fig)
        else:
            raise ValueError("save_path must end with .mp4 or .gif")
    else:
        plt.show()

    return anim

