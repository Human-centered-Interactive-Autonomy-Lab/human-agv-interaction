import pandas as pd
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from visualization.transition_matrix import plot_trust_transition
from visualization.plot_frechet_heatmap import plot_spatiotemporal_trajectories
from visualization.radial_AGV_trust_map import plot_agv_direction_trust
from visualization.plot_3d import plot_3d_response_surface
from visualization.dependent_differences import plot_dependent_differences_by_drate
from visualization.feature_trends import *
from visualization.time_series_trends import visualize
from visualization.expect_heatmap import plot_clustered_heatmaps, plot_heatmap_triplet
from visualization.polar_distribution import plot_polar_distribution
from visualization.agv_clustering import plot_gaze_on_agv
from visualization.agv_in_user_fov import plot_agv_user_fov

from clustering.preprocessing import DataProcessor
from clustering.processing import *
from clustering.post_hoc_analysis import *

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.linear_model import LinearRegression

def main():
    base_dir = os.getcwd()
    print(base_dir)
    data_path = os.path.join(base_dir, 'data')
    
    per_interaction_data = pd.read_csv(os.path.join(data_path, 'per_interaction_data.csv'))
    print(per_interaction_data.head())
    print(per_interaction_data.columns.tolist())

    study_data = pd.read_csv(os.path.join(data_path, 'study_data_by_sec.csv'))
    # processor = DataProcessor(study_data).run_mappings()
    # print(study_data.columns.tolist())
    # print(study_data.head())
    # study_data.to_csv(os.path.join(data_path, 'study_data_by_sec.csv'), index=False)
    
    MFPCA1 = pd.read_csv(os.path.join(data_path, 'MFPCA_scores_with_clusters.csv'))
    MFPCA2 = pd.read_csv(os.path.join(data_path, 'MFPCA_scores_with_clusters_DV.csv'))
    MFPCA3 = pd.read_csv(os.path.join(data_path, 'mfpca_interaction_data.csv'))
    MFPCA4 = pd.read_csv(os.path.join(data_path, 'mfpca_study_data.csv'))


    # fig, axes = plot_agv_user_fov(data=study_data, save_path=None, n_samples=8, fov_half_angle_deg=60, scale_factor=100.0, gaze_arrow_len_m=150.0, random_state=60)


    # interpolate_multivariate_to_common_grid(study_data, 'Timestamp', 'PID', 
    # ['User_X', 'User_Y', 'GazeOrigin_X', 'GazeOrigin_Y', 'GazeDirection_X', 'GazeDirection_Y', 'User_Relative_Speed', 'Angle_to_AGV'], grid_points: int = 365)

    # mfpca_interaction_data = make_mfpca_interaction_data(per_interaction_data, save_path=os.path.join(base_dir, 'data', 'mfpca_interaction_data.csv'))
    # print(mfpca_interaction_data.columns.tolist())
    # print(mfpca_interaction_data.head())

    # mfpca_study_data = make_mfpca_study_data(study_data, mfpca_interaction_data, save_path=os.path.join(base_dir, 'data', 'mfpca_study_data.csv'), before_seconds = 20, after_seconds = 10)
    # mfpca_study_data = loess_smooth_by_pid(
    # mfpca_study_data,
    #     ["Angle_to_AGV", "AGV_User_Distance", "Frechet_Distance", "User_X", "User_Y", "User_Z"],
    #     x_col="Time_Index",  # or "NormTime"
    #     frac=0.05,
    #     save_path=os.path.join(base_dir, 'data', 'mfpca_study_data.csv')
    # )
    # mfpca_study_data = normalize_columns_globally_minmax(mfpca_study_data, ["Angle_to_AGV", "AGV_User_Distance", "Frechet_Distance", "User_X", "User_Y", "User_Z"], save_path=os.path.join(base_dir, 'data', 'mfpca_study_data.csv'))

    # print(mfpca_study_data.columns.tolist())
    # print(mfpca_study_data.head())    

    # plot_gaze_on_agv(study_data, save_path = None)
    plot_combined_mean_DVs_with_clusters(MFPCA3, MFPCA1, save_path=None)
    feature_cols  = ["Angle_to_AGV", "AGV_User_Distance", "Frechet_Distance",
                 "AGV_spd", "User_Relative_Speed", "L1_Deviation",
                 "Cosine_Similarity", "Angle_Deviation"]

    external_cols = ["Trust", "Comfort", "Safe"]  # not used to build clusters
    
    plot_cluster_feature(MFPCA4, MFPCA1, feature="L1_Deviation", save_path = None, smoothing='loess', loess_frac=0.1) # "Angle_to_AGV", "AGV_User_Distance", "Frechet_Distance", "AGV_spd", "User_Relative_Speed", "L1_Deviation", "Cosine_Similarity", "Angle_Deviation"
    plot_combined_mean_features_with_clusters(MFPCA3, MFPCA2, save_path=None)
    # plot_pca_3d(MFPCA1)
    run_anova_and_posthoc(MFPCA2, pc_col="PC2", run_normality=True)

    compute_silhouette_from_pcs(MFPCA2)

    compare_clusterings(MFPCA1, MFPCA2)
    get_aligned_pid_matches(MFPCA1, MFPCA2)
    # plot_polar_distribution(per_interaction_data, MFPCA1, save_path = None)
    # plot_heatmap_triplet(per_interaction_data, var1='Safe', var2='Trust', var3='Comfort', z_col='Expect', save_path=os.path.join(base_dir, 'images', 'Expect_heatmap.png'))
    plot_clustered_heatmaps(per_interaction_data, MFPCA2, var1='Safe', var2='Trust', var3='Comfort', z_col='Expect') # save_path=os.path.join(base_dir, 'images', 'clustered_Expect_heatmap.png'))
    
    
    # polar_distribution(per_interaction_data, clustering_data, save_path=os.path.join(base_dir, 'images', 'clustered_polar_distribution_of_dvs.png'))
    # plot_trust_transition(per_interaction_data, save_path=os.path.join(base_dir, 'images', 'transition_matrices.png'))
    # plot_spatiotemporal_trajectories(study_data, save_path=os.path.join(base_dir, 'images', 'frechet_heatmap.png'))
    # plot_agv_direction_trust(per_interaction_data, trust_col='Safe', direction_col='AGV_Approaching', save_path=os.path.join(base_dir, 'images', 'agv_trust_radial.png'))
    # plot_3d_response_surface(per_interaction_data, x_col='Trust', y_col='Safe', z_col='Expect', title= "Expect Surface", save_path=os.path.join(base_dir, 'images'))
    # plot_dependent_differences_by_drate(per_interaction_data.sort_values(by=['PID', 'StartTime'], ascending=[True, True]))


    # Apply classification
    # trust_summary = compute_trust_patterns(per_interaction_data)
    # trust_labeled = classify_trust_patterns(trust_summary, n_clusters = 2)

    # Extract PIDs by pattern
    # patterns = trust_labeled.groupby('Trust_Pattern')['PID'].apply(list).to_dict()
    # print(patterns)
    
    

    


if __name__ == '__main__':
    main()
