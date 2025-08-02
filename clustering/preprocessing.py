# Once the study data is processed with these columns, delete this file.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

class DataProcessor:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def run_mappings(self):
        self.map_user_trajectory()
        self.map_agv_approaching()
        self.create_agv_user_combination()
        self.compute_angle_and_fov()
        self.add_expected_user_trajectory()
        self.plot_sampled_trajectories()

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

    def create_agv_user_combination(self):
        self.data['AGV_User_Combination'] = (
            self.data['AGV_Approaching'].astype(str) + ' - ' + self.data['User_Trajectory'].astype(str)
        )

    def compute_angle_and_fov(self, fov_degrees: float = 90.0) -> pd.DataFrame:
        """
        Computes the angle between user's gaze and AGV direction, and determines whether the AGV is in FOV.

        Parameters:
        - fov_degrees (float): Field of view angle in degrees (default is 95)

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

            
