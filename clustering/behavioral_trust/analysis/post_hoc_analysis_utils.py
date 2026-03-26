import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, pairwise_distances, adjusted_rand_score, normalized_mutual_info_score
from scipy import stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
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

high_first_low_last = [2, 4, 6, 8, 10, 12, 14, 16, 20, 
                       22, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45]
low_first_high_last = [1, 3, 7, 9, 11, 13, 15, 17, 19, 
                        21, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46]

def run_pre_post_tests_auto(
    df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    *,
    alpha: float = 0.05,
    save_path: str | None = None,
    pid_col: str = "PID",
    cluster_col: str = "Cluster",
    drate_col: str = "DRate",
    timestamp_col: str = "Timestamp",
) -> pd.DataFrame:
    """
    For each DV (pre + post), run:
      - Shapiro normality per cluster
      - Levene homoscedasticity across clusters (reported, not used for Welch decision)
      - If BOTH clusters pass Shapiro (p>=alpha): Welch t-test + Hedges' g
      - Else: Mann–Whitney U + rank-biserial correlation

        TestSets:
            - Pre: one row per PID
            - Post_First: one row per PID×DRate for each PID's first survey condition
            - Post_Second: one row per PID×DRate for each PID's second survey condition
    """

    pre_survey_cols = [
        "PerfectAutomation_high_expectation", "PerfectAutomation_all_or_none",
        "TrustPropensity", "AutomationExperience",
        "AutomationExperience_confidence_mean", "AutomationExperience_reliance_mean",
        "AutomationExperience_trust_mean", "AutomationExperience_safety_mean"
    ]

    post_survey_cols = [
        "MMTrust", "MM_competence", "MM_predictability",
        "MM_dependability", "MM_responsibility", "MM_reliability", "MM_faith",
        "JianTrust", "Jian_distrust", "Jian_trust",
        "MWL", "MWL_performance", "MWL_rest"
    ]

    # ----------------------------
    # Checks + merge clusters
    # ----------------------------
    if pid_col not in df.columns:
        raise ValueError(f"df must contain '{pid_col}'.")
    if drate_col not in df.columns:
        raise ValueError(f"df must contain '{drate_col}'.")
    if timestamp_col not in df.columns:
        df = df.copy()
        df[timestamp_col] = np.arange(len(df), dtype=float)

    df = df.merge(cluster_df[[pid_col, cluster_col]], on=pid_col, how="inner")

    # Determine cluster IDs present (assumes two clusters; works with >2 for reporting,
    # but applied tests are defined for 2 groups only.)
    cluster_ids = sorted(df[cluster_col].dropna().astype(int).unique().tolist())
    if len(cluster_ids) != 2:
        raise ValueError(f"Expected exactly 2 clusters for t/MWU tests; got {cluster_ids}")

    c1, c2 = cluster_ids[0], cluster_ids[1]

    # ----------------------------
    # Helpers
    # ----------------------------
    def _one_row_per_group(d: pd.DataFrame, group_keys: list[str], cols: list[str]) -> pd.DataFrame:
        use_cols = group_keys + cols + [timestamp_col]
        tmp = d[use_cols].copy().sort_values(group_keys + [timestamp_col])

        def _collapse(g: pd.DataFrame) -> pd.Series:
            out = {}
            for col in cols:
                s = g[col]
                out[col] = s.dropna().iloc[-1] if s.notna().any() else np.nan
            return pd.Series(out)

        collapsed = tmp.groupby(group_keys, as_index=False).apply(_collapse, include_groups=False)
        if isinstance(collapsed.index, pd.MultiIndex):
            collapsed = collapsed.reset_index()
        for k in group_keys:
            if k not in collapsed.columns and k in collapsed.index.names:
                collapsed = collapsed.reset_index()
        return collapsed[group_keys + cols]

    def _hedges_g(x: np.ndarray, y: np.ndarray) -> float:
        """Hedges' g for two independent samples."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return np.nan

        vx = np.var(x, ddof=1)
        vy = np.var(y, ddof=1)
        sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
        if sp == 0 or not np.isfinite(sp):
            return np.nan

        d = (np.mean(x) - np.mean(y)) / sp
        # small-sample correction
        J = 1 - (3 / (4 * (nx + ny) - 9))
        return float(J * d)

    def _welch_df(x: np.ndarray, y: np.ndarray) -> float:
        """Welch-Satterthwaite degrees of freedom for two independent samples."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return np.nan

        vx = np.var(x, ddof=1)
        vy = np.var(y, ddof=1)
        term_x = vx / nx
        term_y = vy / ny
        denominator = (term_x ** 2) / (nx - 1) + (term_y ** 2) / (ny - 1)

        if denominator <= 0 or not np.isfinite(denominator):
            return np.nan

        numerator = (term_x + term_y) ** 2
        return float(numerator / denominator)

    def _rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
        """
        Rank-biserial correlation (RBC) derived from Mann–Whitney U:
          RBC = 1 - 2*U/(n1*n2)  (when U corresponds to group1 vs group2)
        Range [-1, 1]. Positive means group1 tends to be larger than group2.
        """
        if n1 <= 0 or n2 <= 0:
            return np.nan
        return float(1 - (2 * u_stat) / (n1 * n2))

    def _run_for_dataset(data_subset: pd.DataFrame, cols: list[str], testset_label: str, out_rows: list[dict]):
        for col in cols:
            row = {"TestSet": testset_label, "Column": col}
            row["T_value"] = np.nan
            row["Degrees_of_Freedom"] = np.nan

            # Values per cluster (numeric only)
            x = pd.to_numeric(data_subset.loc[data_subset[cluster_col] == c1, col], errors="coerce").dropna().to_numpy()
            y = pd.to_numeric(data_subset.loc[data_subset[cluster_col] == c2, col], errors="coerce").dropna().to_numpy()

            row[f"N_C{c1}"] = len(x)
            row[f"N_C{c2}"] = len(y)

            # Shapiro per cluster (needs n>=3)
            for cid, arr in [(c1, x), (c2, y)]:
                if len(arr) >= 3:
                    _, p = shapiro(arr)
                    row[f"Shapiro_p_C{cid}"] = float(p)
                    row[f"Reject_Normality_C{cid}"] = bool(p < alpha)
                else:
                    row[f"Shapiro_p_C{cid}"] = np.nan
                    row[f"Reject_Normality_C{cid}"] = np.nan

            # Levene across clusters (needs n>=2 per group)
            if len(x) >= 2 and len(y) >= 2:
                _, pL = levene(x, y, center="median")
                row["Levene_p"] = float(pL)
                row["Reject_Homoscedasticity"] = bool(pL < alpha)
            else:
                row["Levene_p"] = np.nan
                row["Reject_Homoscedasticity"] = np.nan

            # Decide test based on normality:
            sh1 = row.get(f"Reject_Normality_C{c1}")
            sh2 = row.get(f"Reject_Normality_C{c2}")
            normal_ok = (sh1 is False) and (sh2 is False)

            # If Shapiro couldn't be computed (n<3), be conservative → MWU
            if pd.isna(row[f"Shapiro_p_C{c1}"]) or pd.isna(row[f"Shapiro_p_C{c2}"]):
                normal_ok = False

            if normal_ok:
                # Welch t-test
                t_stat, p = ttest_ind(x, y, equal_var=False, nan_policy="omit")
                row["Applied_Test"] = "Welch_t"
                row["Test_statistic"] = float(t_stat) if np.isfinite(t_stat) else np.nan
                row["T_value"] = float(t_stat) if np.isfinite(t_stat) else np.nan
                row["Degrees_of_Freedom"] = _welch_df(x, y)
                row["Test_p"] = float(p) if np.isfinite(p) else np.nan
                row["Effect_Size_Type"] = "Hedges_g"
                row["Effect_Size"] = _hedges_g(x, y)
            else:
                # Mann–Whitney U (two-sided)
                if len(x) >= 1 and len(y) >= 1:
                    u_stat, p = mannwhitneyu(x, y, alternative="two-sided")
                    row["Applied_Test"] = "MannWhitneyU"
                    row["Test_statistic"] = float(u_stat) if np.isfinite(u_stat) else np.nan
                    row["Test_p"] = float(p) if np.isfinite(p) else np.nan
                    row["Effect_Size_Type"] = "RankBiserial"
                    row["Effect_Size"] = _rank_biserial_from_u(u_stat, len(x), len(y))
                else:
                    row["Applied_Test"] = "MannWhitneyU"
                    row["Test_statistic"] = np.nan
                    row["Test_p"] = np.nan
                    row["Effect_Size_Type"] = "RankBiserial"
                    row["Effect_Size"] = np.nan

            out_rows.append(row)

    # ----------------------------
    # Build datasets and run
    # ----------------------------
    out_rows: list[dict] = []

    have_pre = [c for c in pre_survey_cols if c in df.columns]
    if have_pre:
        pre_base = df[[pid_col, cluster_col] + have_pre + [timestamp_col]].copy()
        pre_one = _one_row_per_group(pre_base, [pid_col, cluster_col], have_pre)
        _run_for_dataset(pre_one, have_pre, "Pre", out_rows)

    have_post = [c for c in post_survey_cols if c in df.columns]
    if have_post:
        post_base = df[[pid_col, drate_col, cluster_col] + have_post + [timestamp_col]].copy()
        post_one = _one_row_per_group(post_base, [pid_col, drate_col, cluster_col], have_post)
        print(f"[DEBUG] Post-survey one-row-per-group shape: {post_one.shape}")

        high_first_set = set(high_first_low_last)
        low_first_set = set(low_first_high_last)

        def _assign_survey_order(row: pd.Series) -> str | None:
            pid = row[pid_col]
            drate = str(row[drate_col]).strip().lower()
            if pid in high_first_set:
                first, second = "high", "low"
            elif pid in low_first_set:
                first, second = "low", "high"
            else:
                return None

            if drate == first:
                return "First"
            if drate == second:
                return "Second"
            return None

        post_one = post_one.copy()
        post_one["Survey_Order"] = post_one.apply(_assign_survey_order, axis=1)

        for order in ["First", "Second"]:
            post_cond = post_one[post_one["Survey_Order"] == order].copy()
            _run_for_dataset(post_cond, have_post, f"Post_{order}", out_rows)

    out = pd.DataFrame(out_rows)

    # ----------------------------
    # Formatting + save
    # ----------------------------
    # Report test outputs with 3 significant figures.
    for c in ["Test_p", "Levene_p", f"Shapiro_p_C{c1}", f"Shapiro_p_C{c2}"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: "" if pd.isna(x) else f"{float(x):.3g}")

    # effect sizes: keep sign, readable
    if "Effect_Size" in out.columns:
        out["Effect_Size"] = out["Effect_Size"].apply(lambda x: "" if pd.isna(x) else f"{float(x):.3g}")

    if "Test_statistic" in out.columns:
        out["Test_statistic"] = out["Test_statistic"].apply(lambda x: "" if pd.isna(x) else f"{float(x):.3g}")

    for c in ["T_value", "Degrees_of_Freedom"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda x: "" if pd.isna(x) else f"{float(x):.3g}")

    if save_path:
        out.to_csv(save_path, index=False)

    return out


def run_anova_with_clusters(
    df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    interaction_col: str = 'Interaction_No',
    save_path: str = None  # e.g., "survey_result.csv"
) -> pd.DataFrame:
    pre_survey_cols = [
        "PerfectAutomation_high_expectation", "PerfectAutomation_all_or_none",
        "TrustPropensity", "AutomationExperience",
        "AutomationExperience_confidence_mean", "AutomationExperience_reliance_mean",
        "AutomationExperience_trust_mean", "AutomationExperience_safety_mean"
    ]
    post_survey_cols = [
        "MMTrust", "MM_competence", "MM_predictability",
        "MM_dependability", "MM_responsibility", "MM_reliability", "MM_faith",
        "JianTrust", "Jian_distrust", "Jian_trust",
        "MWL", "MWL_performance", "MWL_rest"
    ]

    # Merge clusters
    df = df.merge(cluster_df[['PID', 'Cluster']], on='PID', how='inner')

    # All cluster IDs present
    cluster_ids = sorted(df['Cluster'].dropna().astype(int).unique().tolist())

    results = []

    # --- helper to add one row (col/test) with p-value and cluster means ---
    def _append_result_row(data_subset: pd.DataFrame, col: str, test_label: str):
        if data_subset.empty:
            return
        # Fit ANOVA (rows with NaN in col or Cluster will be dropped automatically)
        model = smf.ols(f"{col} ~ C(Cluster)", data=data_subset).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        # robust fetch of p-value
        pval = float(anova_table.loc['C(Cluster)', 'PR(>F)'])

        # Cluster-wise means using the same rows used in the model
        used = model.model.data.frame  # rows actually used by statsmodels
        means = used.groupby('Cluster')[col].mean()

        row = {'Column': col, 'Test': test_label, 'p-value': pval}
        # add dynamic mean columns: Mean_C<id>
        for cid in cluster_ids:
            row[f"Mean_C{cid}"] = means.get(cid, np.nan)
        results.append(row)

    # --- Pre-Survey (one row per PID) ---
    if pre_survey_cols:
        pre_df = df[['PID', 'Cluster'] + pre_survey_cols].drop_duplicates('PID')
        for col in pre_survey_cols:
            _append_result_row(pre_df[['Cluster', col]].copy(), col, 'Pre')

    # --- Post-Survey: FirstHalf (1–16) and SecondHalf (17–32) ---
    if post_survey_cols:
        halves = [('Post_FirstHalf', 1, 16), ('Post_SecondHalf', 17, 32)]
        for label, lo, hi in halves:
            post_df = df[(df[interaction_col] >= lo) & (df[interaction_col] <= hi)]
            for col in post_survey_cols:
                _append_result_row(post_df[['Cluster', col, interaction_col]].copy(), col, label)

    # Create DataFrame
    survey_result = pd.DataFrame(results)

    # Holm correction over all tests
    corrected = multipletests(survey_result['p-value'].astype(float), method='holm')
    survey_result['Corrected_p'] = corrected[1]
    survey_result['Reject_H0'] = corrected[0]

    # Format p-values (3 sig figs) and means (2 sig figs)
    survey_result['p-value'] = survey_result['p-value'].apply(lambda x: f"{x:.4g}")
    survey_result['Corrected_p'] = survey_result['Corrected_p'].apply(lambda x: f"{x:.4g}")

    mean_cols = [c for c in survey_result.columns if c.startswith('Mean_C')]
    for c in mean_cols:
        survey_result[c] = survey_result[c].apply(lambda x: ("" if pd.isna(x) else f"{float(x):.4g}"))

    # Optional export
    if save_path:
        survey_result.to_csv(save_path, index=False)

    # Also return for immediate use
    return survey_result



