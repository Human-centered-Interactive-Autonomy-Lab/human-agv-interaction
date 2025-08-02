import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, pairwise_distances, adjusted_rand_score, normalized_mutual_info_score
from statsmodels.stats.multitest import multipletests
from scipy import stats
from scipy.stats import levene, shapiro
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import Sequence, Union
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

def _merge_on_pid(df: pd.DataFrame, cluster_df: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(cluster_df[['PID', 'Cluster']], on='PID', how='inner').copy()
    if out['Cluster'].isna().any():
        out = out.dropna(subset=['Cluster'])
    return out


def _cohens_d(a, b):
    na, nb = len(a), len(b)
    s2a, s2b = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * s2a + (nb - 1) * s2b) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def _dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    """Dunn = min inter-cluster dist / max intra-cluster dist."""
    D = pairwise_distances(X)
    labs = np.unique(labels)
    # max intra
    intra_max = 0.0
    for k in labs:
        idx = np.where(labels == k)[0]
        if len(idx) > 1:
            intra_max = max(intra_max, D[np.ix_(idx, idx)].max())
    # min inter
    inter_min = np.inf
    for i, j in combinations(labs, 2):
        ii = np.where(labels == i)[0]
        jj = np.where(labels == j)[0]
        inter_min = min(inter_min, D[np.ix_(ii, jj)].min())
    return inter_min / intra_max if intra_max > 0 else np.nan


def compute_effect_sizes(features_df: pd.DataFrame,
                         cluster_df: pd.DataFrame,
                         feature_cols,
                         aggregate_per_pid: bool = True,
                         agg_func: str = "mean") -> pd.DataFrame:
    """
    Returns pairwise Cohen's d for each feature between clusters.

    Parameters
    ----------
    features_df : DataFrame (must contain PID + feature_cols)
    cluster_df  : DataFrame with ['PID', 'Cluster']
    feature_cols: list[str]
    aggregate_per_pid : bool
        If True, first aggregate repeated rows per PID via agg_func.
    agg_func : str
        'mean', 'median', etc.

    Returns
    -------
    DataFrame with columns: feature, cluster_a, cluster_b, cohens_d
    """
    df = _merge_on_pid(features_df, cluster_df)

    if aggregate_per_pid:
        df = df.groupby(['PID', 'Cluster'], as_index=False)[feature_cols].agg(agg_func)

    clusters = df['Cluster'].unique()
    results = []

    if len(clusters) < 2:
        raise ValueError("Need at least two clusters to compute effect sizes.")

    for f in feature_cols:
        for ca, cb in combinations(clusters, 2):
            a = df.loc[df['Cluster'] == ca, f].dropna()
            b = df.loc[df['Cluster'] == cb, f].dropna()
            d = _cohens_d(a, b)
            results.append((f, ca, cb, d))

    return pd.DataFrame(results, columns=['feature', 'cluster_a', 'cluster_b', 'cohens_d'])


def compute_internal_indices(features_df: pd.DataFrame,
                             cluster_df: pd.DataFrame,
                             feature_cols,
                             standardize: bool = True,
                             aggregate_per_pid: bool = True,
                             agg_func: str = "mean") -> dict:
    """
    Returns silhouette, Dunn, and Calinski–Harabasz indices.

    Parameters
    ----------
    features_df : DataFrame (must contain PID + feature_cols)
    cluster_df  : DataFrame with ['PID', 'Cluster']
    feature_cols: list[str]
    standardize : bool
        Standardize features before computing indices.
    aggregate_per_pid : bool
        Aggregate repeated rows per PID by agg_func.

    Returns
    -------
    dict with {'silhouette', 'dunn', 'calinski_harabasz'}
    """
    df = _merge_on_pid(features_df, cluster_df)

    if aggregate_per_pid:
        df = df.groupby(['PID', 'Cluster'], as_index=False)[feature_cols].agg(agg_func)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df['Cluster'].to_numpy()

    if standardize:
        X = StandardScaler().fit_transform(X)

    # Need at least 2 clusters and > 1 sample
    if len(np.unique(y)) < 2 or len(y) < 3:
        return {'silhouette': np.nan, 'dunn': np.nan, 'calinski_harabasz': np.nan}

    silhouette = silhouette_score(X, y)
    ch = calinski_harabasz_score(X, y)
    dunn = _dunn_index(X, y)

    return {'silhouette': silhouette, 'dunn': dunn, 'calinski_harabasz': ch}

def compute_bw_dispersion_ratio(features_df: pd.DataFrame,
                                cluster_df: pd.DataFrame,
                                feature_cols,
                                standardize: bool = True,
                                aggregate_per_pid: bool = True,
                                agg_func: str = "mean") -> float:
    """
    Returns the between/within dispersion ratio.

    Parameters
    ----------
    features_df : DataFrame (must contain PID + feature_cols)
    cluster_df  : DataFrame with ['PID', 'Cluster']
    feature_cols: list[str]
    standardize : bool
    aggregate_per_pid : bool
    agg_func : str

    Returns
    -------
    float (np.nan if within == 0)
    """
    df = _merge_on_pid(features_df, cluster_df)

    if aggregate_per_pid:
        df = df.groupby(['PID', 'Cluster'], as_index=False)[feature_cols].agg(agg_func)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df['Cluster'].to_numpy()

    if standardize:
        X = StandardScaler().fit_transform(X)

    overall_mean = X.mean(axis=0)
    clusters = np.unique(y)

    between = 0.0
    within = 0.0
    for c in clusters:
        Xc = X[y == c]
        if Xc.size == 0:
            continue
        mean_c = Xc.mean(axis=0)
        between += len(Xc) * np.sum((mean_c - overall_mean) ** 2)
        within  += np.sum((Xc - mean_c) ** 2)

    return between / within if within > 0 else np.nan



def test_external_variables(external_df: pd.DataFrame,
                            cluster_df: pd.DataFrame,
                            external_cols,
                            aggregate_per_pid: bool = True,
                            agg_func: str = "mean",
                            test: str = "ttest",
                            p_adjust: str = "fdr_bh") -> pd.DataFrame:
    """
    Runs two-sample tests on variables NOT used in clustering.

    Parameters
    ----------
    external_df : DataFrame (must contain PID + external_cols)
    cluster_df  : DataFrame with ['PID', 'Cluster']
    external_cols : list[str]
    aggregate_per_pid : bool
    agg_func : str
    test : {'ttest', 'mannwhitney'}
    p_adjust : str
        Any method supported by statsmodels.stats.multitest.multipletests

    Returns
    -------
    DataFrame with test stats, raw & adjusted p-values, and Cohen's d.
    """
    df = _merge_on_pid(external_df, cluster_df)

    if aggregate_per_pid:
        df = df.groupby(['PID', 'Cluster'], as_index=False)[external_cols].agg(agg_func)

    clusters = df['Cluster'].unique()
    if len(clusters) != 2:
        raise ValueError("This helper currently assumes exactly 2 clusters for testing.")

    ca, cb = sorted(clusters)
    results = []
    for col in external_cols:
        a = df.loc[df['Cluster'] == ca, col].dropna()
        b = df.loc[df['Cluster'] == cb, col].dropna()

        if test == "ttest":
            stat, p = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
        elif test == "mannwhitney":
            stat, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        else:
            raise ValueError("test must be 'ttest' or 'mannwhitney'.")

        d = _cohens_d(a, b)
        results.append((col, stat, p, d))

    out = pd.DataFrame(results, columns=["variable", "stat", "p_raw", "cohens_d"])
    if p_adjust:
        out["p_adj"] = multipletests(out["p_raw"], method=p_adjust)[1]
    return out.sort_values("p_adj" if p_adjust else "p_raw")


def run_anova_and_posthoc(data: pd.DataFrame, pc_col: str, cluster_col: str = "Cluster", run_normality=False):
    """
    Performs Levene's test, one-way ANOVA, and Tukey HSD post-hoc on a principal component grouped by clusters.

    Parameters:
    - data: DataFrame containing the PC scores and cluster labels.
    - pc_col: Name of the principal component column to analyze (e.g., 'PC1').
    - cluster_col: Name of the cluster label column (default is 'Cluster').
    - run_normality: Whether to perform Shapiro-Wilk test for normality per group (default False).
    """
    df = data[[pc_col, cluster_col]].dropna()

    # Levene’s test for homogeneity of variances
    grouped = [group[pc_col].values for _, group in df.groupby(cluster_col)]
    lev_stat, lev_p = levene(*grouped)
    print(f"Levene's Test for Equal Variances:\n  Statistic = {lev_stat:.4f}, p-value = {lev_p:.4f}")
    if lev_p > 0.05:
        print("  Variance homogeneity assumed.")
    else:
        print("  Variance homogeneity violated. Consider using Welch ANOVA or robust tests.")

    # Shapiro-Wilk test for normality (optional)
    if run_normality:
        print("\nShapiro-Wilk Test for Normality by Cluster:")
        for name, group in df.groupby(cluster_col):
            stat, p = shapiro(group[pc_col])
            print(f"  Cluster {name}: Statistic = {stat:.4f}, p-value = {p:.4f}")

    # One-way ANOVA
    model = ols(f'{pc_col} ~ C({cluster_col})', data=df).fit()
    anova_results = anova_lm(model, typ=2)
    print("\nANOVA Results:")
    print(anova_results)

    # Post-hoc test (only if ANOVA is significant)
    if anova_results["PR(>F)"].iloc[0] < 0.05:
        print("\nTukey HSD Post-hoc Test:")
        tukey_result = pairwise_tukeyhsd(endog=df[pc_col], groups=df[cluster_col], alpha=0.05)
        print(tukey_result.summary())
    else:
        print("\nNo significant group differences found. Post-hoc test not performed.")

def compute_silhouette_from_pcs(
    cluster_df: pd.DataFrame,
    pc_cols: Sequence[str] = ("PC1", "PC2", "PC3"),
    cluster_col: str = "Cluster",
    standardize: bool = True,
    metric: str = "euclidean"
    ):
    """
    Compute the silhouette score from principal components stored in `cluster_df`.

    Parameters
    ----------
    cluster_df : pd.DataFrame
        Must contain the columns in `pc_cols` and `cluster_col`.
    pc_cols : sequence of str
        The PC columns to use as features.
    cluster_col : str
        Column with cluster labels.
    standardize : bool
        If True, standardize PCs before computing the silhouette.
    metric : str
        Distance metric passed to sklearn.metrics.silhouette_score.

    Returns
    -------
    float or np.nan
        Silhouette score, or np.nan if it cannot be computed
        (e.g., fewer than 2 clusters or too few samples).
    """
    # Drop rows with any NA in PCs or the cluster label
    needed = list(pc_cols) + [cluster_col]
    df = cluster_df.dropna(subset=needed).copy()
    if df.empty:
        return np.nan

    X = df.loc[:, pc_cols].to_numpy(dtype=float)
    labels = df[cluster_col].to_numpy()

    # Need at least 2 clusters and >= 2 samples total
    unique_clusters = np.unique(labels)
    if len(unique_clusters) < 2 or X.shape[0] < 2:
        return np.nan

    if standardize:
        X = StandardScaler().fit_transform(X)

    # Also ensure every cluster has at least 2 samples (silhouette requirement)
    counts = pd.Series(labels).value_counts()
    if (counts < 2).any():
        return np.nan

    print("Silhouette score:", silhouette_score(X, labels, metric=metric))

def compare_clusterings(df1, df2):
    # Merge on PID
    merged = pd.merge(df1, df2, on='PID', suffixes=('_1', '_2'))
    
    # Correct column names
    labels1 = merged['Cluster_1']
    labels2 = merged['Cluster_2']
    
    # Compute metrics
    ari = adjusted_rand_score(labels1, labels2)
    nmi = normalized_mutual_info_score(labels1, labels2)
    
    result =  {
        'Adjusted Rand Index': ari,
        'Normalized Mutual Information': nmi,
        'Num Common PIDs': len(merged)
    }
    print(result)

def get_aligned_pid_matches(df1, df2):
    # Merge both cluster assignments
    merged = pd.merge(df1, df2, on='PID', suffixes=('_1', '_2'))

    labels1 = merged['Cluster_1'].values
    labels2 = merged['Cluster_2'].values

    # Build contingency matrix
    contingency = confusion_matrix(labels1, labels2)
    
    # Hungarian algorithm to align labels
    row_ind, col_ind = linear_sum_assignment(-contingency)
    
    # Create mapping from Cluster_2 to aligned Cluster_1
    label_map = {col: row for row, col in zip(row_ind, col_ind)}
    merged['Aligned_Cluster_2'] = merged['Cluster_2'].map(label_map)

    # Find matching PIDs
    matching_pids = merged[merged['Cluster_1'] == merged['Aligned_Cluster_2']]['PID'].tolist()
    
    print(f"PIDs clustered the same way in both algorithms: {matching_pids}")


