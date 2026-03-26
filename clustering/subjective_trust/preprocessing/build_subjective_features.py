import pandas as pd
import numpy as np
from pathlib import Path
import os
from scipy.stats import skew, mode
from scipy.stats import linregress

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

def make_subjective_clustering_data(
    data: pd.DataFrame,
    save_path: str | Path | None = None,
    *,
    base_vars: list[str] | None = None,
    pid_col: str = "PID",
    time_col: str = "StartTime",
    interaction_col: str = "Interaction_No",
    fill_missing_interactions: bool = True,
    interaction_range: tuple[int, int] = (1, 32),
    add_rolling_stats: bool = True,
    add_interaction_differences: bool = True,
    skew_eps: float = 1e-8,
) -> pd.DataFrame:
    
    if base_vars is None:
        base_vars = ["Trust", "Expect", "Safe", "Comfort"]

    df = data.copy()

    # --- checks ---
    if pid_col not in df.columns:
        raise ValueError(f"Missing required column '{pid_col}'.")
    for v in base_vars:
        if v not in df.columns:
            raise ValueError(f"Variable '{v}' not found in dataframe.")

    # --- stable ordering ---
    if time_col in df.columns:
        df = df.sort_values([pid_col, time_col]).copy()
    else:
        df = df.sort_values([pid_col]).copy()

    # --- ensure Interaction_No exists ---
    if interaction_col not in df.columns:
        df = df.groupby(pid_col, group_keys=False).apply(assign_interaction_no)

    # make Interaction_No numeric-int where possible
    df[interaction_col] = pd.to_numeric(df[interaction_col], errors="coerce").astype("Int64")

    # --- optionally fill missing interactions per PID (1..32) ---
    if fill_missing_interactions:
        start_i, end_i = interaction_range
        full = pd.DataFrame({interaction_col: np.arange(start_i, end_i + 1, dtype=int)})

        all_cols = df.columns.tolist()
        filled = []
        for pid, g in df.groupby(pid_col, sort=False):
            g2 = g.merge(full, on=interaction_col, how="right")
            g2[pid_col] = pid
            g2 = g2.reindex(columns=all_cols)
            filled.append(g2)

        df = (
            pd.concat(filled, ignore_index=True)
              .sort_values([pid_col, interaction_col])
              .reset_index(drop=True)
        )
    else:
        df = df.sort_values([pid_col, interaction_col]).reset_index(drop=True)

    # after filling, Interaction_No might be floaty; keep it int-like where possible
    df[interaction_col] = pd.to_numeric(df[interaction_col], errors="coerce").astype("Int64")

    # ------------------------------------------------------------------
    # A) Rolling/cumulative stats (slow-ish but matches your original logic)
    # ------------------------------------------------------------------
    if add_rolling_stats:
        # initialize columns
        for var in base_vars:
            df[f"{var}_Mean"] = np.nan
            df[f"{var}_Slope"] = np.nan
            df[f"{var}_STD"] = np.nan
            df[f"{var}_Range"] = np.nan
            df[f"{var}_Mode"] = np.nan
            df[f"{var}_Median"] = np.nan
            df[f"{var}_Skewness"] = np.nan

        # compute per PID in interaction order
        for pid, group in df.groupby(pid_col, sort=False):
            # group is already sorted by Interaction_No
            idxs = group.index.to_numpy()

            for var in base_vars:
                values = group[var].to_numpy(dtype=float)

                for i in range(len(values)):
                    slice_ = values[: i + 1]
                    slice_clean = slice_[~np.isnan(slice_)]
                    times = np.arange(1, len(slice_clean) + 1)

                    if len(slice_clean) == 0:
                        continue

                    df.loc[idxs[i], f"{var}_Mean"] = float(np.mean(slice_clean))
                    df.loc[idxs[i], f"{var}_STD"] = float(np.std(slice_clean, ddof=0))
                    df.loc[idxs[i], f"{var}_Range"] = float(np.max(slice_clean) - np.min(slice_clean))
                    df.loc[idxs[i], f"{var}_Median"] = float(np.median(slice_clean))

                    # mode: safe even for len=1
                    df.loc[idxs[i], f"{var}_Mode"] = float(mode(slice_clean, keepdims=True).mode[0])

                    # skewness: undefined/unstable if nearly constant or too few points
                    if len(slice_clean) < 3 or np.std(slice_clean) < skew_eps:
                        df.loc[idxs[i], f"{var}_Skewness"] = np.nan
                    else:
                        df.loc[idxs[i], f"{var}_Skewness"] = float(skew(slice_clean, bias=True))

                    # slope: needs >=2 points
                    if len(slice_clean) < 2:
                        df.loc[idxs[i], f"{var}_Slope"] = 0.0
                    else:
                        df.loc[idxs[i], f"{var}_Slope"] = float(linregress(times, slice_clean).slope)

    # ------------------------------------------------------------------
    # B) Interaction differences (fast, vectorized)
    # ------------------------------------------------------------------
    if add_interaction_differences:
        gb = df.groupby(pid_col, sort=False)
        for var in base_vars:
            prev = gb[var].shift(1)
            nxt = gb[var].shift(-1)

            df[f"{var}_Diff_Before"] = df[var] - prev
            df[f"{var}_Diff_After"] = nxt - df[var]

    # ensure Interaction_No is plain int if you want (will be NaN if missing)
    # df[interaction_col] = df[interaction_col].astype("Int64")

    # --- save ---
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)

    return df
