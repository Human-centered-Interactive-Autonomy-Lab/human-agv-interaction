import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import os

# Sample placeholder for the function
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
