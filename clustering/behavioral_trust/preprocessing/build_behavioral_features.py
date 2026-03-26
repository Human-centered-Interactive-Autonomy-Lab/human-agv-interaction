import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from numpy.linalg import norm

class DataProcessor:
    def __init__(self, data: pd.DataFrame, eye_data: pd.DataFrame):
        self.data = data.sort_values(by=["PID", "Timestamp"]).reset_index(drop=True)
        self.eye_data = eye_data.sort_values(by=["PID", "Timestamp"]).reset_index(drop=True)

    def run_mappings(self):
        self.map_user_trajectory()
        self.map_agv_approaching()
        self.map_eye_target()
        self.create_agv_user_combination()
        self.compute_angle_and_fov()
        self.add_expected_user_trajectory()
        self.plot_sampled_trajectories()
        self.compute_user_speed_and_heading()
        self.compute_total_distance_travelled()
        self.compute_orientation_changes()
        self.compute_velocity_components()
        self.compute_gaze_vector_angle_to_agv()
        self.compute_gaze_instability()
        self.compute_proximity_to_agv()
        self.compute_time_to_collision()  
        self.compute_gaze_on_agv_time()
        self.compute_user_attention_state()
        return self

    def map_user_trajectory(self):
        user_trajectory_mapping = {
            1: 'Straight', 2: 'Diagonal', 3: 'Diagonal', 4: 'Straight', 5: 'Straight',
            6: 'Diagonal', 7: 'Straight', 8: 'Diagonal', 9: 'Diagonal', 10: 'Straight',
            11: 'Diagonal', 12: 'Straight', 13: 'Straight', 14: 'Diagonal', 15: 'Straight',
            16: 'Diagonal'
        }
        self.data['User_Trajectory'] = self.data['AGVname'].map(user_trajectory_mapping)

    def map_agv_approaching(self):
        agv_approaching_mapping = {
            1: 'South', 2: 'North', 3: 'South', 4: 'Northeast', 5: 'Northwest', 
            6: 'Northwest', 7: 'East', 8: 'Southwest', 9: 'Northeast', 10: 'West', 
            11: 'East', 12: 'Southeast', 13: 'Southwest', 14: 'Southeast', 15: 'North', 
            16: 'West'
        }
        self.data['AGV_Approaching'] = self.data['AGVname'].map(agv_approaching_mapping)

    def map_eye_target(self):
        # Harmonize dtypes on join keys
        for df in (self.data, self.eye_data):
            df['PID'] = pd.to_numeric(df['PID'], errors='coerce')
            df['DRate'] = df['DRate'].astype(str).str.strip()

        # Keep one instance of each unique (PID, DRate, Timestamp, Eye_Target)
        k4 = ['PID', 'DRate', 'Timestamp', 'Eye_Target']
        eye_unique = (
            self.eye_data
            .dropna(subset=['PID', 'DRate', 'Timestamp'])
            .drop_duplicates(subset=k4, keep='last')
        )

        # Remove any existing Eye_Target and merge
        self.data.drop(columns=['Eye_Target'], inplace=True, errors='ignore')
        self.data = self.data.merge(
            eye_unique[['PID', 'DRate', 'Timestamp', 'Eye_Target']],
            on=['PID', 'DRate', 'Timestamp'],
            how='left'
        )

        # Boolean flag from Eye_Target
        self.data['Gaze_on_AGV'] = self.data['Eye_Target'].astype('string').str.contains('AGV', case=False, na=False)
        # Convert to 0 and 1
        self.data['Gaze_on_AGV'] = self.data['Gaze_on_AGV'].astype(int)

    def create_agv_user_combination(self):
        self.data['AGV_User_Combination'] = (
            self.data['AGV_Approaching'].astype(str) + ' - ' + self.data['User_Trajectory'].astype(str)
        )

    def compute_angle_and_fov(self, fov_degrees: float = 45.0) -> pd.DataFrame:
        """
        Computes the angle between user's gaze and AGV direction, and determines whether the AGV is in FOV.

        Parameters:
        - fov_degrees (float): Field of view angle in degrees (default is 45) because of Meta quests given FOV.

        Returns:
        - pd.DataFrame with two new columns:
            - 'Angle_to_AGV': angle in degrees between gaze and AGV direction [0, 360)
            - 'AGV_in_FOV': boolean, True if AGV is in user's FOV
        """
        # Vectors from user to AGV
        vec_user_to_agv = np.stack([
            self.data['AGV_X'] - self.data['User_X'],
            self.data['AGV_Y'] - self.data['User_Y'],
            self.data['AGV_Z'] - self.data['User_Z'],
        ], axis=1)

        # User gaze direction
        gaze_direction = np.stack([
            self.data['GazeDirection_X'],
            self.data['GazeDirection_Y'],
            self.data['GazeDirection_Z']
        ], axis=1)

        def normalize(vectors):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            return vectors / np.where(norms == 0, 1, norms)  # Prevent division by zero

        # Normalize both vectors
        vec_user_to_agv_norm = normalize(vec_user_to_agv)
        gaze_direction_norm = normalize(gaze_direction)

        # Compute angles relative to +X axis
        theta_agv = np.arctan2(vec_user_to_agv_norm[:, 1], vec_user_to_agv_norm[:, 0])
        theta_gaze = np.arctan2(gaze_direction_norm[:, 1], gaze_direction_norm[:, 0])

        # Angle difference
        angle_diff = (theta_gaze - theta_agv + 2 * np.pi) % (2 * np.pi)
        angles_deg = np.degrees(angle_diff)

        # Determine FOV
        half_fov = fov_degrees / 2
        in_fov = (angles_deg <= half_fov) | (angles_deg >= (360 - half_fov))

        result = pd.DataFrame({
            'Angle_to_AGV': angles_deg,
            'AGV_in_FOV': in_fov,
            'Gaze_on_AGV': abs(angles_deg) < 5
        })

        self.data[['Angle_to_AGV', 'AGV_in_FOV', 'Gaze_on_AGV']] = result

    def add_expected_user_trajectory(self) -> pd.DataFrame:
        """
        Adds Expected_User_X and Expected_User_Y columns to the dataframe based on straight-line interpolation
        from first to last User_X/Y within each (PID, DRate, AGVname) group.
        """

        # Ensure columns exist
        if not {'PID', 'DRate', 'AGVname', 'User_X', 'User_Y'}.issubset(self.data.columns):
            raise ValueError("The dataframe must contain 'PID', 'DRate', 'AGVname', 'User_X', and 'User_Y' columns.")

        # Initialize empty columns
        self.data['Expected_User_X'] = np.nan
        self.data['Expected_User_Y'] = np.nan

        # Group by the specified keys
        grouped = self.data.groupby(['PID', 'DRate', 'AGVname'])

        # Apply straight-line interpolation for each group
        for (pid, drate, agvname), group in grouped:
            idxs = group.index
            n = len(group)

            # Get start and end coordinates
            start_x, start_y = group.iloc[0][['User_X', 'User_Y']]
            end_x, end_y = group.iloc[-1][['User_X', 'User_Y']]

            # Linear interpolation
            interp_x = np.linspace(start_x, end_x, n)
            interp_y = np.linspace(start_y, end_y, n)

            # Assign to original DataFrame
            self.data.loc[idxs, 'Expected_User_X'] = interp_x
            self.data.loc[idxs, 'Expected_User_Y'] = interp_y

    def plot_sampled_trajectories(self, n_samples=4, seed=42):
        """
        Randomly selects n_samples groups and plots actual vs. expected user trajectories.
        """
        # Ensure expected columns exist
        if not {'Expected_User_X', 'Expected_User_Y'}.issubset(self.data.columns):
            raise ValueError("Expected columns not found. Please run 'add_expected_user_trajectory' first.")

        # Get unique groups
        group_keys = list(self.data.groupby(['PID', 'DRate', 'AGVname']).groups.keys())
        random.seed(seed)
        sampled_keys = random.sample(group_keys, min(n_samples, len(group_keys)))

        # Plot
        plt.figure(figsize=(16, 12))

        for i, key in enumerate(sampled_keys, 1):
            sub_data = self.data[(self.data['PID'] == key[0]) & (self.data['DRate'] == key[1]) & (self.data['AGVname'] == key[2])]

            plt.subplot(2, 2, i)
            plt.plot(sub_data['User_X'], sub_data['User_Y'], 'o-', label='Actual Trajectory', alpha=0.7)
            plt.plot(sub_data['Expected_User_X'], sub_data['Expected_User_Y'], 'x--', label='Expected Trajectory', alpha=0.7)
            plt.title(f'PID: {key[0]}, DRate: {key[1]}, AGVname: {key[2]}')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.legend()
            plt.axis('equal')

        plt.tight_layout()
        plt.show()

    def compute_user_speed_and_heading(self):
        def compute(group):
            pos = group[['User_X', 'User_Y', 'User_Z']].values
            delta = np.diff(pos, axis=0, prepend=np.nan)
            displacement = np.linalg.norm(delta, axis=1)
            
            # Compute heading vector
            heading = [vec / norm(vec) if norm(vec) != 0 else np.array([np.nan]*3) for vec in delta]
            
            # Assign speed and heading
            group['User_Speed'] = displacement
            group[['Heading_X', 'Heading_Y', 'Heading_Z']] = pd.DataFrame(heading, index=group.index)

            # Compute acceleration (difference of speed)
            group['User_Acceleration'] = group['User_Speed'].diff().fillna(0)

            return group

        self.data = self.data.groupby('PID', group_keys=False).apply(compute)

    def compute_total_distance_travelled(self):
        def compute(group):
            group['Distance_Travelled'] = np.cumsum(np.nan_to_num(group['User_Speed']))
            return group
        self.data = self.data.groupby('PID', group_keys=False).apply(compute)
        #print(self.data.columns.tolist())

    def compute_orientation_changes(self):
        def compute(group):
            group['Yaw_Change'] = group['User_Yaw'].diff().abs()
            group['Rotation_Change'] = np.sqrt(group['User_Pitch'].diff()**2 +
                                                group['User_Yaw'].diff()**2 +
                                                group['User_Roll'].diff()**2)
            return group
        self.data = self.data.groupby('PID', group_keys=False).apply(compute)
        #print(self.data.columns.tolist())

    def compute_velocity_components(self):
        self.data['Velocity_Vector'] = self.data[['U_X', 'U_Y', 'U_Z']].values.tolist()
        self.data['Speed_X'] = self.data['U_X']
        self.data['Speed_Y'] = self.data['U_Y']
        self.data['Speed_Z'] = self.data['U_Z']
        #print(self.data.columns.tolist())

    def compute_gaze_vector_angle_to_agv(self):
        gaze_vector = self.data[['GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z']].values
        agv_pos = self.data[['AGV_X', 'AGV_Y', 'AGV_Z']].values
        gaze_origin = self.data[['GazeOrigin_X', 'GazeOrigin_Y', 'GazeOrigin_Z']].values
        vector_to_agv = agv_pos - gaze_origin
        dot = np.einsum('ij,ij->i', gaze_vector, vector_to_agv)
        gaze_norm = norm(gaze_vector, axis=1)
        target_norm = norm(vector_to_agv, axis=1)
        cos_theta = dot / (gaze_norm * target_norm)
        self.data['Gaze_Angle_to_AGV'] = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        #print(self.data.columns.tolist())

    def compute_gaze_instability(self):
        gaze_vector = self.data[['GazeDirection_X', 'GazeDirection_Y', 'GazeDirection_Z']].values
        self.data['Gaze_Instability'] = np.insert(np.linalg.norm(np.diff(gaze_vector, axis=0), axis=1), 0, np.nan)
        #print(self.data.columns.tolist())

    def compute_proximity_to_agv(self):
        self.data['Proximity_to_AGV'] = norm(
            self.data[['User_X', 'User_Y', 'User_Z']].values -
            self.data[['AGV_X', 'AGV_Y', 'AGV_Z']].values, axis=1
        )
        #print(self.data.columns.tolist())

    def compute_gaze_on_agv_time(self):
        self.data['Cumulative_Gaze_on_AGV'] = self.data.groupby('PID')['Gaze_on_AGV'].cumsum()
        self.data['Avg_Gaze_on_AGV'] = self.data['Cumulative_Gaze_on_AGV'] / (
        self.data.groupby('PID').cumcount() + 1
        )
        #print(self.data.columns.tolist())

    def compute_user_attention_state(self):
        self.data['Paused'] = self.data['User_Speed'] < 0.01
        self.data['Start_Stop'] = self.data['Paused'].astype(int).diff().abs().fillna(0)
        #print(self.data.columns.tolist())

    def compute_time_to_collision(self) -> pd.DataFrame:
        """
        Adds 'time_to_collision' at each timestamp, computed within each (PID, AGVname, DRate) group.

        TTC is computed using line-of-sight closing speed:
            r = p_agv - p_user
            v = v_agv - v_user
            closing_speed = -(r · v) / ||r||
            TTC = ||r|| / closing_speed   if closing_speed > 0
                 inf                      otherwise (not closing)

        Assumes point-mass collision (distance -> 0). If you want collision radius, we can modify.
        """
        print("In the time to collision function.")
        required = {'PID', 'AGVname', 'DRate', 'Timestamp',
                    'User_X', 'User_Y', 'User_Z',
                    'AGV_X', 'AGV_Y', 'AGV_Z'}
        missing = required - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns for TTC: {sorted(missing)}")

        def _dt_seconds(ts: pd.Series) -> pd.Series:
            """
            Robust dt (seconds) from Timestamp.
            Supports:
              - datetime64 / pandas Timestamp
              - timedelta64 / pandas Timedelta
              - numeric (assumed seconds)
              - strings parseable as datetime or timedelta
            """
            s = ts

            # If already datetime-like
            if np.issubdtype(s.dtype, np.datetime64):
                dt = s.diff().dt.total_seconds()
                return dt

            # If already timedelta-like
            if np.issubdtype(s.dtype, np.timedelta64):
                dt = s.diff().dt.total_seconds()
                return dt

            # Try parse as datetime
            s_dt = pd.to_datetime(s, errors='coerce', utc=False)
            if s_dt.notna().mean() > 0.9:
                return s_dt.diff().dt.total_seconds()

            # Try parse as timedelta (e.g., "0 days 00:00:00.100000")
            s_td = pd.to_timedelta(s, errors='coerce')
            if s_td.notna().mean() > 0.9:
                return s_td.diff().dt.total_seconds()

            # Fallback: numeric seconds
            s_num = pd.to_numeric(s, errors='coerce')
            return s_num.diff()

        def _compute_group(g: pd.DataFrame) -> pd.DataFrame:
            g = g.sort_values('Timestamp').copy()

            dt = _dt_seconds(g['Timestamp']).astype(float)
            # Avoid division by 0 / negative dt; keep NaN where dt invalid
            dt = dt.where(dt > 0, np.nan)

            # Positions
            u_pos = g[['User_X', 'User_Y', 'User_Z']].to_numpy(dtype=float)
            a_pos = g[['AGV_X', 'AGV_Y', 'AGV_Z']].to_numpy(dtype=float)

            # Velocities estimated from position differences / dt
            u_delta = np.vstack([np.full((1, 3), np.nan), np.diff(u_pos, axis=0)])
            a_delta = np.vstack([np.full((1, 3), np.nan), np.diff(a_pos, axis=0)])

            dt_arr = dt.to_numpy(dtype=float).reshape(-1, 1)
            u_vel = u_delta / dt_arr
            a_vel = a_delta / dt_arr

            # Relative position / velocity
            r = a_pos - u_pos
            v = a_vel - u_vel

            dist = np.linalg.norm(r, axis=1)
            # Prevent divide-by-zero when user and AGV positions coincide
            dist_safe = np.where(dist == 0, np.nan, dist)

            # closing_speed = -(r·v)/||r||
            rv = np.einsum('ij,ij->i', r, v)
            closing_speed = -(rv / dist_safe)

            # TTC: distance / closing_speed if closing_speed > 0, else inf
            ttc = np.full(len(g), np.inf, dtype=float)
            valid = np.isfinite(dist) & np.isfinite(closing_speed) & (closing_speed > 0)

            ttc[valid] = dist[valid] / closing_speed[valid]

            # First row has no velocity estimate -> set NaN instead of inf for cleanliness
            ttc[0] = np.nan

            g['Time_to_Collision'] = ttc
            return g

        self.data = (
            self.data
            .groupby(['PID', 'AGVname', 'DRate'], group_keys=False)
            .apply(_compute_group)
            .reset_index(drop=True)
        )