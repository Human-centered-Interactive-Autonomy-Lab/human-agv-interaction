import pandas as pd
import os
from visualization.trust_heatmap import plot_trust_heatmap
from visualization.transition_matrix import plot_trust_transition
from visualization.plot_frechet_heatmap import plot_spatiotemporal_trajectories
from visualization.radial_AGV_trust_map import plot_agv_direction_trust
from visualization.plot_3d import plot_3d_response_surface
from visualization.coordinates import plot_coordinates




def main():
    # Load Data
    base_dir = os.getcwd()
    print(base_dir)
    data_path = os.path.join(base_dir, 'data')
    
    per_interaction_data = pd.read_csv(os.path.join(data_path, 'per_interaction_data.csv'))
    study_data = pd.read_csv(os.path.join(data_path, 'study_data_processed.csv'))

    print(per_interaction_data.head())
    print(per_interaction_data.columns.tolist())
    # plot_trust_heatmap(per_interaction_data, var1='Safe', var2='Expect', var3='Comfort', z_col='Trust', save_path=os.path.join(base_dir, 'images'))
    # plot_trust_transition(per_interaction_data, save_path=os.path.join(base_dir, 'images', 'transition_matrices.png'))
    plot_spatiotemporal_trajectories(study_data, save_path=os.path.join(base_dir, 'images', 'frechet_heatmap.png'))
    # plot_agv_direction_trust(per_interaction_data, trust_col='Safe', direction_col='AGV_Approaching', save_path=os.path.join(base_dir, 'images', 'agv_trust_radial.png'))
    # plot_3d_response_surface(per_interaction_data, x_col='Trust', y_col='Safe', z_col='Expect', title= "Expect Surface", save_path=os.path.join(base_dir, 'images'))
    # plot_coordinates(study_data, save_path=os.path.join(base_dir, 'images'))


if __name__ == '__main__':
    main()
