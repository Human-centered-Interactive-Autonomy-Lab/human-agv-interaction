import pandas as pd
from pathlib import Path
import os

from visual_analysis_utils import *

def main():
    processed_data_dir = os.path.join(os.path.join(os.getcwd(), "data"), "processed")
    processed_dir = Path(processed_data_dir)

    raw_data_dir = os.path.join(os.path.join(os.getcwd(), "data"), "raw")
    raw_dir = Path(raw_data_dir)
    
    behavioral_clustering_assignments = pd.read_csv(os.path.join(processed_dir, 'cluster_assignments_behavioral_kmeans.csv'))
    subjective_clustering_data = pd.read_csv(os.path.join(processed_dir, 'subjective_clustering_data.csv'))
    behavioral_clustering_data = pd.read_csv(os.path.join(processed_dir, 'behavioral_clustering_data.csv'))
    per_second_data = pd.read_csv(os.path.join(raw_dir, 'per_second_data.csv'))

    # Behavioral clustering process visulization
    columns_to_visualize = ['Gaze_Angle_to_AGV', 'User_Speed', 'Gaze_Instability','Frechet_Distance']
    save_fig_path=Path.cwd() / "clustering" / "behavioral_trust" /"figures" / "analysis"/ "cluster_profiles"
    # plot_all_cluster_features(behavioral_clustering_data, behavioral_clustering_assignments, feature_cols=columns_to_visualize, smoothing='loess', loess_frac=0.0, ci = 1.96, save_path = os.path.join(save_fig_path, 'behavioral_feature_trajectories_normalized_by_bc.png')) # os.path.join(save_fig_path, 'behavioral_feature_trajectories_normalized_by_bc.png'))
    # plot_other_features(behavioral_clustering_data, behavioral_clustering_assignments, smoothing='loess', loess_frac=0.1, save_path = os.path.join(save_fig_path, 'other_gaze_features.png')) # save_path = os.path.join(base_dir, 'images', 'other_gaze_features.png')
    # plot_combined_mean_DVs_with_clusters(subjective_clustering_data, behavioral_clustering_assignments , save_path=os.path.join(save_fig_path, 'subjective_mean_dvs_by_bc.png'))
    # plot_efficiency_vs_safety(subjective_clustering_data, behavioral_clustering_assignments, per_second_data, save_path=os.path.join(save_fig_path, 'subjective_efficiency_vs_safety_by_bc.png'))
    plot_categorical_survey_responses(subjective_clustering_data, behavioral_clustering_assignments, save_path=os.path.join(save_fig_path, 'pre_survey_responses_by_bc.png'))
    # plot_cross_first_by_cluster(subjective_clustering_data, behavioral_clustering_assignments, save_path=os.path.join(save_fig_path, 'right_of_way_by_bc.png'))
    # plot_cross_first_between_clusters(subjective_clustering_data, behavioral_clustering_assignments, save_path=os.path.join(save_fig_path, 'right-of-way_decisions_across_bc.png'))
    # animate_user_interactions_with_fov(
    # behavioral_clustering_data, pid=3,
    # save_path=os.path.join(save_fig_path, 'cluster_1_representative.gif'), 
    # fps=7,
    # divide_by=100.0,
    # xlim=(0, 175), ylim=(0, 150),  # match your plant bounds if you want fixed axes
    # fov_half_angle_deg=45,
    # gaze_arrow_len_units=50.0
    #) 
    

if __name__ == '__main__':
    main()
