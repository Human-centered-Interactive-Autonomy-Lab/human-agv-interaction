import pandas as pd
import numpy as np
from scipy.stats import skew, mode
from scipy.stats import linregress
import os
from sklearn.preprocessing import MinMaxScaler
from statsmodels.nonparametric.smoothers_lowess import lowess

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

def transform_angle(angle):
    if pd.isna(angle):
        return np.nan
    return 360 - angle if angle > 180 else angle

def make_mfpca_interaction_data(data: pd.DataFrame, save_path=None) -> pd.DataFrame:
    data = data.copy()
    data = data.sort_values(['PID', 'StartTime'])
    data = data.groupby('PID', group_keys=False).apply(assign_interaction_no)

    # Variables to process
    base_vars = ['Trust', 'Expect', 'Safe', 'Comfort']

    # Add new columns for each variable
    for var in base_vars:
        data[f'{var}_Mean'] = np.nan
        data[f'{var}_Slope'] = np.nan
        data[f'{var}_STD'] = np.nan
        data[f'{var}_Range'] = np.nan
        data[f'{var}_Mode'] = np.nan
        data[f'{var}_Median'] = np.nan
        data[f'{var}_Skewness'] = np.nan

    # Compute rolling stats per PID per variable
    for pid, group in data.groupby('PID'):
        for var in base_vars:
            values = group[var].values
            for i in range(len(values)):
                slice_ = values[:i+1]
                times = np.arange(1, len(slice_)+1)

                data.loc[group.index[i], f'{var}_Mean'] = np.mean(slice_)
                data.loc[group.index[i], f'{var}_STD'] = np.std(slice_)
                data.loc[group.index[i], f'{var}_Range'] = np.max(slice_) - np.min(slice_)
                data.loc[group.index[i], f'{var}_Mode'] = mode(slice_, keepdims=True).mode[0] if len(slice_) > 0 else np.nan
                data.loc[group.index[i], f'{var}_Median'] = np.median(slice_)
                data.loc[group.index[i], f'{var}_Skewness'] = skew(slice_)
                data.loc[group.index[i], f'{var}_Slope'] = linregress(times, slice_).slope if len(slice_) > 1 else 0

    # Ensure Interaction_No is integer
    data['Interaction_No'] = data['Interaction_No'].astype(int)

    # Fill in missing Interaction_No from 1 to 32 for each PID
    complete_rows = []
    all_cols = data.columns

    for pid, group in data.groupby('PID'):
        full_range = pd.DataFrame({'Interaction_No': np.arange(1, 33)})
        group = group.merge(full_range, on='Interaction_No', how='right')
        group['PID'] = pid
        # Sort and reorder columns
        group = group[[col for col in all_cols if col in group.columns]]
        complete_rows.append(group)

    data = pd.concat(complete_rows).sort_values(by=['PID', 'Interaction_No']).reset_index(drop=True)

    # Save if path is specified
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path, index=False)

    return data


def _update_expected_and_frechet(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("Time_Index").copy()
    n = len(g)
    if n == 0:
        return g

    # If X/Y missing, just return NAs
    if g[['User_X', 'User_Y']].isna().any().any():
        for col in ["Expected_User_X", "Expected_User_Y", "Frechet_Distance",
                    "L1_Deviation", "Cosine_Similarity", "Angle_Deviation"]:
            g[col] = np.nan
        return g

    # 1. Expected coordinates
    x0, y0 = g[['User_X', 'User_Y']].iloc[0].to_numpy()
    x1, y1 = g[['User_X', 'User_Y']].iloc[-1].to_numpy()
    t = np.linspace(0.0, 1.0, n)
    g['Expected_User_X'] = x0 + (x1 - x0) * t
    g['Expected_User_Y'] = y0 + (y1 - y0) * t

    # 2. Fréchet distance (discrete 2-point)
    frechet = np.full(n, np.nan)
    l1_dev = np.full(n, np.nan)
    cos_sim = np.full(n, np.nan)
    angle_dev = np.full(n, np.nan)

    act_xy = g[['User_X', 'User_Y']].to_numpy()
    exp_xy = g[['Expected_User_X', 'Expected_User_Y']].to_numpy()

    for i in range(1, n):
        p0, p1 = exp_xy[i-1], exp_xy[i]
        q0, q1 = act_xy[i-1], act_xy[i]

        # Fréchet = max of endpoint distances
        frechet[i] = max(np.linalg.norm(p0 - q0), np.linalg.norm(p1 - q1))

        # L1 norm at current step
        l1_dev[i] = np.abs(act_xy[i] - exp_xy[i]).sum()

        # Cosine similarity between expected & actual velocity vectors
        v_expected = p1 - p0
        v_actual = q1 - q0

        norm_e = np.linalg.norm(v_expected)
        norm_a = np.linalg.norm(v_actual)

        if norm_e > 0 and norm_a > 0:
            cos_theta = np.dot(v_expected, v_actual) / (norm_e * norm_a)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            cos_sim[i] = cos_theta
            angle_dev[i] = np.degrees(np.arccos(cos_theta))
        else:
            cos_sim[i] = np.nan
            angle_dev[i] = np.nan

    g['Frechet_Distance'] = frechet
    g['L1_Deviation'] = l1_dev
    g['Cosine_Similarity'] = cos_sim
    g['Angle_Deviation'] = angle_dev

    return g

def make_mfpca_study_data(
    study_data: pd.DataFrame,
    mfpca_data: pd.DataFrame,
    save_path: str = None,
    before_seconds: int = 10,
    after_seconds: int = 10
    ) -> pd.DataFrame:
    """
    Constructs a time-windowed dataset around the minimum AGV_User_Distance point
    for each interaction, with configurable before/after windows.

    Parameters:
    - study_data: Main dataset containing time series info
    - mfpca_data: Metadata to map interactions
    - save_path: Optional path to save final DataFrame as CSV
    - before_seconds: Seconds before interaction to include
    - after_seconds: Seconds after interaction to include

    Returns:
    - DataFrame containing time-aligned, per-interaction data
    """
    # Sort by PID and Timestamp
    study_data = study_data.sort_values(by=['PID', 'Timestamp']).copy()

    # Merge to assign Interaction_No using PID, DRate, AGVname
    merged = pd.merge(
        study_data,
        mfpca_data[['PID', 'DRate', 'AGVname', 'Interaction_No']],
        on=['PID', 'DRate', 'AGVname'],
        how='left'
    )

    # Step 1: Extract time windows around interaction events
    windows = []
    grouped = merged.groupby(['PID', 'DRate', 'AGVname', 'Interaction_No'], dropna=False)
    for keys, group in grouped:
        if group['Interaction_No'].isna().all():
            continue
        ts_min = pd.to_datetime(group.loc[group['AGV_User_Distance'].idxmin(), 'Timestamp'])
        group['Timestamp'] = pd.to_datetime(group['Timestamp'])
        ts_window = (
            ts_min - pd.Timedelta(seconds=before_seconds),
            ts_min + pd.Timedelta(seconds=after_seconds)
        )
        window = group[(group['Timestamp'] >= ts_window[0]) & (group['Timestamp'] <= ts_window[1])].copy()
        windows.append(window)

    if not windows:
        return pd.DataFrame()  # nothing matched

    result = pd.concat(windows, ignore_index=True)

    # Step 2: Handle missing interactions
    dummy_length = before_seconds + after_seconds + 1
    full_rows = []
    for pid in result['PID'].unique():
        existing_interactions = result[result['PID'] == pid]['Interaction_No'].dropna().unique()
        missing = [i for i in range(1, 33) if i not in existing_interactions]
        for missing_i in missing:
            dummy = pd.DataFrame({
                'PID': [pid] * dummy_length,
                'Interaction_No': [missing_i] * dummy_length,
                'Normalized_Time': list(range(1, dummy_length + 1))
            })
            full_rows.append(dummy)

    # Step 3: Assign Normalized Time and append dummy rows
    result['Normalized_Time'] = result.groupby(['PID', 'Interaction_No']).cumcount() + 1
    if full_rows:
        missing_df = pd.concat(full_rows, ignore_index=True)
        result = pd.concat([result, missing_df], ignore_index=True)

    # Step 4: Sort and index
    result = result.sort_values(by=['PID', 'Interaction_No', 'Normalized_Time']).reset_index(drop=True)
    result['Time_Index'] = result.groupby('PID').cumcount() + 1

    # Step 5: Angle transformation (if applicable)
    if 'Angle_to_AGV' in result.columns:
        result['Angle_to_AGV'] = result['Angle_to_AGV'].apply(transform_angle)

    # Step 6: Add Expected_* and Frechet_Distance
    if {'User_X', 'User_Y'}.issubset(result.columns):
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
    - A new DataFrame with normalized columns appended with '_norm'
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