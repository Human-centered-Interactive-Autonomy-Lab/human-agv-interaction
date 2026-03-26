import pandas as pd
import numpy as np

import os
from sklearn.preprocessing import MinMaxScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.spatial.distance import euclidean
from tqdm import tqdm
import seaborn as sns

def assign_interaction_no(group):
    # Define the PID groups
    high_first_low_last = [2, 4, 6, 8, 10, 12, 14, 16, 20, 
                       22, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45]

    low_first_high_last = [1, 3, 7, 9, 11, 13, 15, 17, 19, 
                        21, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46]

    pid = group['PID'].iloc[0]
    is_high_first = pid in high_first_low_last

    interaction_no = []

    for _, row in group.iterrows():
        agv = row['AGVname']
        drate = row['DRate']
        if (is_high_first and drate == 'High') or (not is_high_first and drate == 'Low'):
            interaction_no.append(agv)
        else:
            interaction_no.append(agv + 16)
    group['Interaction_No'] = interaction_no
    return group

def add_interaction_numbers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["PID", "StartTime"])
    df = df.groupby("PID", group_keys=False).apply(assign_interaction_no)
    df["Interaction_No"] = df["Interaction_No"].astype(int)
    return df

def transform_angle(angle):
    if pd.isna(angle):
        return np.nan
    return 360 - angle if angle > 180 else angle

def discrete_frechet(P, Q):
    """Compute discrete Fréchet distance between two curves P and Q."""
    n, m = len(P), len(Q)
    ca = np.full((n, m), -1.0)

    def c(i, j):
        if ca[i, j] > -1:
            return ca[i, j]
        d = euclidean(P[i], Q[j])
        if i == 0 and j == 0:
            ca[i, j] = d
        elif i == 0:
            ca[i, j] = max(c(0, j-1), d)
        elif j == 0:
            ca[i, j] = max(c(i-1, 0), d)
        else:
            ca[i, j] = max(min(c(i-1, j), c(i-1, j-1), c(i, j-1)), d)
        return ca[i, j]

    return c(n-1, m-1)

'''
def _update_expected_and_frechet(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("Time_Index").copy()
    n = len(g)
    if n == 0:
        return g

    # If any user position is missing, return NaNs for outputs
    if g[['User_X', 'User_Y']].isna().any().any():
        g['Expected_User_X'] = np.nan
        g['Expected_User_Y'] = np.nan
        g['Frechet_Distance'] = np.nan
        return g

    # Expected straight-line path from first to last
    x0, y0 = g[['User_X', 'User_Y']].iloc[0]
    x1, y1 = g[['User_X', 'User_Y']].iloc[-1]
    t = np.linspace(0.0, 1.0, n)
    g['Expected_User_X'] = x0 + (x1 - x0) * t
    g['Expected_User_Y'] = y0 + (y1 - y0) * t

    # Pointwise Euclidean distance at each step
    act_xy = g[['User_X', 'User_Y']].to_numpy(dtype=float)
    exp_xy = g[['Expected_User_X', 'Expected_User_Y']].to_numpy(dtype=float)
    d = np.linalg.norm(act_xy - exp_xy, axis=1)  # shape (n,)

    # Cumulative max up to time t
    frechet = np.maximum.accumulate(d)

    # If you want the first entry to be NaN (to match your previous behavior):
    frechet[0] = np.nan

    g['Frechet_Distance'] = frechet
    return g

'''

def _update_expected_and_frechet(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("Time_Index").copy()
    n = len(g)
    if n == 0:
        return g

    # Ensure output columns exist
    for col in ["Expected_User_X", "Expected_User_Y",
                "Frechet_Distance", "L1_Deviation",
                "Cosine_Similarity", "Angle_Deviation"]:
        if col not in g.columns:
            g[col] = np.nan

    # Validity mask for actual positions
    valid = g[['User_X', 'User_Y']].notna().all(axis=1)
    valid_idx = np.flatnonzero(valid.to_numpy())

    # Need at least two valid points to define a straight expected path
    if valid_idx.size < 2:
        # Leave the initialized NaNs in place
        return g

    i0, i1 = int(valid_idx[0]), int(valid_idx[-1])
    x0, y0 = g.loc[g.index[i0], ['User_X', 'User_Y']].astype(float)
    x1, y1 = g.loc[g.index[i1], ['User_X', 'User_Y']].astype(float)

    # Build expected straight line across the whole group length (like original)
    t = np.linspace(0.0, 1.0, n, dtype=float)
    g['Expected_User_X'] = x0 + (x1 - x0) * t
    g['Expected_User_Y'] = y0 + (y1 - y0) * t

    # Prepare arrays
    frechet = np.full(n, np.nan, dtype=float)
    l1_dev  = np.full(n, np.nan, dtype=float)
    cos_sim = np.full(n, np.nan, dtype=float)
    ang_dev = np.full(n, np.nan, dtype=float)

    act_xy = g[['User_X', 'User_Y']].to_numpy(dtype=float)
    exp_xy = g[['Expected_User_X', 'Expected_User_Y']].to_numpy(dtype=float)

    # Helper: validity per index (actual AND expected are finite)
    finite = np.isfinite(act_xy).all(axis=1) & np.isfinite(exp_xy).all(axis=1)

    for i in range(1, n):
        # L1 deviation at i (needs current point valid)
        if finite[i]:
            l1_dev[i] = np.abs(act_xy[i] - exp_xy[i]).sum()

        # Directional similarity (needs i and i-1 both valid)
        if finite[i] and finite[i-1]:
            v_e = exp_xy[i] - exp_xy[i-1]
            v_a = act_xy[i] - act_xy[i-1]
            ne = np.linalg.norm(v_e)
            na = np.linalg.norm(v_a)
            if ne > 0.0 and na > 0.0:
                c = float(np.dot(v_e, v_a) / (ne * na))
                c = np.clip(c, -1.0, 1.0)
                cos_sim[i] = c
                ang_dev[i] = np.degrees(np.arccos(c))

        # Fréchet distance up to i:
        # build subsequences using ONLY indices ≤ i that are finite
        mask_up_to_i = finite & (np.arange(n) <= i)
        idx_seq = np.flatnonzero(mask_up_to_i)
        if idx_seq.size >= 2:
            exp_seq = exp_xy[idx_seq]
            act_seq = act_xy[idx_seq]
            try:
                frechet[i] = discrete_frechet(exp_seq, act_seq)
            except Exception:
                # If the routine fails on edge cases, leave NaN and continue
                pass

    g['Frechet_Distance'] = frechet
    g['L1_Deviation']     = l1_dev
    g['Cosine_Similarity']= cos_sim
    g['Angle_Deviation']  = ang_dev

    return g


def apply_with_progress(df, group_keys, func, desc="Processing"):
    results = []
    grouped = df.groupby(group_keys)
    for key, group in tqdm(grouped, total=grouped.ngroups, desc=desc):
        results.append(func(group))
    return pd.concat(results, ignore_index=True)

def make_behavioral_clustering_data(
    study_data: pd.DataFrame,
    mfpca_data: pd.DataFrame,
    save_path: str = None,
    before_seconds: int = 10,
    after_seconds: int = 10
) -> pd.DataFrame:
    """
    Align each interaction on a fixed 1 Hz grid centered at the minimum AGV_User_Distance
    timestamp ("interaction center"). Output has exactly (before + 1 + after) rows per
    (PID, Interaction_No), with NaNs inserted where seconds are missing. Also pads
    completely missing interactions (1..32) for each PID with all-NaN rows (except keys).
    """
    # --- Prep & merge to get Interaction_No on each row ---
    study = study_data.copy()
    study['Timestamp'] = pd.to_datetime(study['Timestamp'])
    study = study.sort_values(['PID', 'Timestamp'])

    meta = mfpca_data[['PID', 'DRate', 'AGVname', 'Interaction_No']].drop_duplicates()
    merged = pd.merge(
        study,
        meta,
        on=['PID', 'DRate', 'AGVname'],
        how='left'
    )

    # Fixed relative second index: -before .. 0 .. +after
    rel_index = np.arange(-before_seconds, after_seconds + 1, dtype=int)
    win_len = len(rel_index)

    aligned_windows = []

    # Group by interaction keys
    grouped = merged.groupby(['PID', 'DRate', 'AGVname', 'Interaction_No'], dropna=False)
    for (pid, drate, agv, inter_no), g in grouped:
        # Skip groups without an Interaction_No
        if pd.isna(inter_no):
            continue

        # If AGV_User_Distance has all NaNs, we can't locate a center — leave entire window NaN.
        # Otherwise find the timestamp of the minimum distance.
        if g['AGV_User_Distance'].notna().any():
            ts_center = g.loc[g['AGV_User_Distance'].idxmin(), 'Timestamp']
        else:
            ts_center = None

        # Compute relative seconds to the (rounded) nearest second around center
        if ts_center is not None:
            g = g.copy()
            rel_sec = (g['Timestamp'] - ts_center).dt.total_seconds().round().astype('Int64')
            g['__rel_sec__'] = rel_sec

            # Keep only rows within the requested window
            g = g[g['__rel_sec__'].between(-before_seconds, after_seconds, inclusive='both')]

            # If multiple rows hit the same second, keep the earliest
            g = g.sort_values(['__rel_sec__', 'Timestamp']).drop_duplicates(subset='__rel_sec__', keep='first')

            # Reindex to the full second grid to create NaN rows where missing
            g = g.set_index('__rel_sec__').reindex(rel_index)
        else:
            # No center available — create a fully-NaN window for this interaction
            g = pd.DataFrame(index=rel_index, columns=merged.columns)

        # Ensure key columns are present (fill with the group keys)
        g['PID'] = pid
        g['DRate'] = drate
        g['AGVname'] = agv
        g['Interaction_No'] = inter_no

        # Helpful columns
        g['Rel_Sec'] = g.index.values
        g['Normalized_Time'] = g['Rel_Sec'] + before_seconds + 1  # 1..win_len
        g['Is_Center'] = (g['Rel_Sec'] == 0).astype(int)

        aligned_windows.append(g.reset_index(drop=True))

    if not aligned_windows:
        result = pd.DataFrame(columns=list(study.columns) + ['Interaction_No', 'Rel_Sec', 'Normalized_Time', 'Is_Center'])
        if save_path:
            result.to_csv(save_path, index=False)
        return result

    result = pd.concat(aligned_windows, ignore_index=True)

    # --- Pad completely missing interactions (1..32) per PID with all-NaN data rows ---
    full_rows = []
    for pid, sub in result.groupby('PID'):
        existing = set(sub['Interaction_No'].dropna().astype(int).unique().tolist())
        missing_interactions = [i for i in range(1, 33) if i not in existing]
        for inter_no in missing_interactions:
            dummy = pd.DataFrame({
                'PID': pid,
                'DRate': np.nan,
                'AGVname': np.nan,
                'Interaction_No': inter_no,
                'Rel_Sec': rel_index,
                'Normalized_Time': np.arange(1, win_len + 1, dtype=int),
                'Is_Center': (rel_index == 0).astype(int)
            })
            # Add all columns that exist in `result`, filling with NaN
            for col in result.columns:
                if col not in dummy.columns:
                    dummy[col] = np.nan
            # Order columns like result
            dummy = dummy[result.columns]
            full_rows.append(dummy)

    if full_rows:
        result = pd.concat([result, *full_rows], ignore_index=True)

    # --- Sort & add a per-PID running Time_Index ---
    result = result.sort_values(['PID', 'Interaction_No', 'Normalized_Time']).reset_index(drop=True)
    result['Time_Index'] = result.groupby('PID').cumcount() + 1

    # --- Optional angle transform ---
    if 'Angle_to_AGV' in result.columns:
        # expects a user-defined function: transform_angle(angle: float) -> float
        result['Angle_to_AGV'] = result['Angle_to_AGV'].apply(transform_angle)

    # --- Optional expected trajectory + Frechet update ---
    if {'User_X', 'User_Y'}.issubset(result.columns):
        # expects a user-defined function: _update_expected_and_frechet(df) -> df
        # MUST handle internal NaNs gracefully.
        result = (
            result
            .groupby(['PID', 'Interaction_No'], group_keys=False)
            .apply(_update_expected_and_frechet)
            .reset_index(drop=True)
        )
    else:
        raise ValueError("The dataframe must contain 'User_X' and 'User_Y' columns.")

    if save_path:
        result.to_csv(save_path, index=False)

    return result

def preprocess_and_truncate_to_min_length(df: pd.DataFrame, save_path=None) -> pd.DataFrame:
    # Ensure Timestamp is datetime
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])

    # Step 1: Fill missing seconds per user
    all_users = []
    for pid, group in df.groupby('PID'):
        group = group.sort_values('Timestamp').copy()
        full_range = pd.date_range(start=group['Timestamp'].min(), end=group['Timestamp'].max(), freq='1S')
        group = group.set_index('Timestamp')
        group = group.reindex(full_range)
        group['PID'] = pid
        group.index.name = 'Timestamp'
        all_users.append(group.reset_index())

    # Combine all reindexed groups
    df_full = pd.concat(all_users, ignore_index=True)

    # Step 2: Find minimum length of available rows per PID
    lengths = df_full.groupby('PID').size()
    min_len = lengths.min()

    # Step 3: Truncate each group to min_len
    truncated = (
        df_full
        .groupby('PID', group_keys=False)
        .apply(lambda g: g.iloc[:min_len])
        .reset_index(drop=True)
    )

    # Step 4: Assign Time_Index
    truncated['Time_Index'] = truncated.groupby('PID').cumcount() + 1
    truncated['Angle_to_AGV'] = truncated['Angle_to_AGV'].apply(transform_angle)

    if save_path:
        truncated.to_csv(save_path, index=False)

    return truncated

def loess_smooth_by_pid(df: pd.DataFrame, columns_to_smooth: list, x_col: str = "Time_Index", frac: float = 0.1, save_path: str = None) -> pd.DataFrame:
    """
    Apply LOESS smoothing to specified columns per PID.

    Parameters:
    - df: Input DataFrame with a 'PID' column and time column (e.g., 'Time_Index')
    - columns_to_smooth: List of feature columns to LOESS smooth
    - x_col: The independent variable column (usually time-based, like 'Time_Index' or 'NormTime')
    - frac: Fraction of data used in each local regression (controls smoothness)
    - save_path: Optional path to save smoothed DataFrame as CSV

    Returns:
    - DataFrame with new columns <col>_loess for each smoothed column
    """
    smoothed_df = df.copy()

    for col in columns_to_smooth:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

        smoothed_col = f"{col}"

        smoothed_df[smoothed_col] = (
            df.groupby("PID", group_keys=False)
              .apply(lambda group: pd.Series(
                  lowess(group[col], group[x_col], frac=frac, return_sorted=False),
                  index=group.index))
        )

    if save_path:
        smoothed_df.to_csv(save_path, index=False)

    return smoothed_df

def normalize_columns_globally_minmax(df: pd.DataFrame, columns_to_normalize: list, save_path=None) -> pd.DataFrame:
    """
    Normalize selected columns globally using MinMax scaling (not by group).

    Parameters:
    - df: Input DataFrame
    - columns_to_normalize: List of column names to normalize

    Returns:
    - A new DataFrame with normalized columns
    """
    normalized_df = df.copy()

    for col in columns_to_normalize:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

        norm_col = f"{col}"
        scaler = MinMaxScaler()
        values = df[[col]].values  # 2D for scaler
        scaled = scaler.fit_transform(values)
        normalized_df[norm_col] = scaled.flatten()

    if save_path:
        normalized_df.to_csv(save_path, index=False)

    return normalized_df


def explore_correlations_and_redundancy(df: pd.DataFrame, columns_to_visualize: list, save_path=None):
    X = df[columns_to_visualize]
    sns.heatmap(pd.DataFrame(X, columns=columns_to_visualize).corr(), annot=True, cmap='coolwarm')
