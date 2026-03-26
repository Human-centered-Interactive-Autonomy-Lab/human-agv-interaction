import pandas as pd
import os
import logging
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from build_behavioral_features import *
from prepare_behavioral_clustering_data import *

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger = logging.getLogger(__name__)

    base_dir = os.getcwd()
    logger.info("Base directory: %s", base_dir)

    data_dir = os.path.join(base_dir, "data")
    logger.info("Data directory: %s", data_dir)

    raw_data_dir = os.path.join(data_dir, "raw")
    logger.info("Raw data directory: %s", raw_data_dir)

    processed_data_dir = os.path.join(data_dir, "processed")
    logger.info("Processed data directory: %s", processed_data_dir)
    
    per_interaction_data = pd.read_csv(os.path.join(raw_data_dir, 'per_interaction_data.csv'))
    eye_data = pd.read_csv(os.path.join(raw_data_dir, 'eye_target_data.csv'))
    per_second_data = pd.read_csv(os.path.join(raw_data_dir, 'per_second_data.csv'))
    # pre_survey = pd.read_csv(os.path.join(data_path, 'pre_survey.csv'))


    processor = DataProcessor(per_second_data, eye_data).run_mappings() 
    per_second_data = processor.data

    per_second_data.to_csv(os.path.join(raw_data_dir, 'per_second_data.csv'), index = False)


    feature_cols  = ['Gaze_Angle_to_AGV', 'User_Speed', 'Gaze_Instability', 'Frechet_Distance', 'User_Acceleration', 'Time_to_Collision']
    per_interaction_indexed = add_interaction_numbers(per_interaction_data)
    behavioral_clustering_data = make_behavioral_clustering_data(
    per_second_data,
    per_interaction_indexed,
    before_seconds=20,
    after_seconds=10,
    save_path=os.path.join(processed_data_dir, "behavioral_clustering_data.csv"),
    )

    behavioral_clustering_data = loess_smooth_by_pid(
        behavioral_clustering_data,
        feature_cols,
        x_col="Time_Index",
        frac=0.15,
        save_path=os.path.join(processed_data_dir, "behavioral_clustering_data.csv"),
    )

    behavioral_clustering_data = normalize_columns_globally_minmax(
        behavioral_clustering_data,
        feature_cols,
        save_path=os.path.join(processed_data_dir, "behavioral_clustering_data.csv"),
    )


    # Correlation Matrix of all the derived features
    

    # MFPCA1 = pd.read_csv(os.path.join(data_path, 'MFPCA_scores_with_clusters_kmeans.csv'))
    # MFPCA2 = pd.read_csv(os.path.join(data_path, 'MFPCA_scores_with_clusters_DV.csv'))
    # MFPCA3 = pd.read_csv(os.path.join(data_path, 'mfpca_interaction_data.csv'))
    # MFPCA4 = pd.read_csv(os.path.join(data_path, 'mfpca_study_data.csv'))
    # print(MFPCA4['PID'].value_counts())

    # Behavioral clustering process visulization
    # plot_agv_user_fov(data=study_data, save_path=os.path.join(base_dir, 'images', 'AGV_in_User_Field_of_View_Visualization.png'), n_samples=2, fov_half_angle_deg=45, scale_factor=100.0, gaze_arrow_len_m=120.0, random_state=47)
    
    columns_to_visualize = ['Gaze_Angle_to_AGV', 'User_Speed', 'Gaze_Instability', 'Frechet_Distance']
    # plot_all_cluster_features(MFPCA4, MFPCA1, feature_cols=columns_to_visualize, smoothing='loess', loess_frac=0.0, ci = 1.96, save_path = os.path.join(base_dir, 'images', 'cluster-wise_normalized_trajectories.png'))
    # plot_other_gaze_features(MFPCA4, MFPCA1, smoothing='loess', loess_frac=0.1, save_path = os.path.join(base_dir, 'images', 'other_gaze_features.png')) # save_path = os.path.join(base_dir, 'images', 'other_gaze_features.png')
    # plot_combined_mean_DVs_with_clusters(MFPCA3, MFPCA1 , save_path=os.path.join(base_dir, 'images', 'mean_DVs_with_clusters.png'))
    # plot_efficiency_vs_safety(MFPCA3, MFPCA1, study_data, save_path=os.path.join(base_dir, 'images', 'efficiency_vs_safety.png'))
    # plot_categorical_survey_responses(MFPCA3, MFPCA1, save_path=os.path.join(base_dir, 'images', 'pre_survey_by_cluster.png'))
    # run_anova_and_posthoc(MFPCA1, pc_col="PC1", run_normality=True)
    # compute_silhouette_from_pcs(MFPCA2)
    # run_anova_with_clusters(MFPCA3, MFPCA1, save_path=os.path.join(base_dir, 'data', 'survey_results.csv'))
    #animate_user_interactions_with_fov(
    # MFPCA4, pid=3,
    # save_path=os.path.join(base_dir, 'images', 'cluster_1_representative.gif'), 
    # fps=7,
    # divide_by=100.0,
    # xlim=(0, 175), ylim=(0, 150),  # match your plant bounds if you want fixed axes
    # fov_half_angle_deg=45,
    # gaze_arrow_len_units=50.0
    #) 
    # summary = plot_cross_first_by_cluster(MFPCA1, MFPCA3)
    
    # Subjective clustering process visulization
    # plot_combined_mean_DVs_with_clusters(MFPCA3, MFPCA2, save_path=os.path.join(base_dir, 'images', 'mean_DVs_with_clusters.png'))
    # plot_all_cluster_features(MFPCA4, MFPCA2, feature_cols=feature_cols, save_path = os.path.join(base_dir, 'images', 'cluster-wise_normalized_trajectories_backward.png'), smoothing='loess', loess_frac=0.05, ci = 1.9, direction_arrows = direction_arrows) 
    # plot_combined_mean_features_with_clusters(MFPCA3, MFPCA2, save_path=None)
    # plot_subjective_bar_comparison(MFPCA3, MFPCA2, save_path=os.path.join(base_dir, 'images', 'subjective_measures_by_cluster_backward.png'))
    # run_anova_with_clusters(MFPCA3, MFPCA2)

    # compare_clusterings(MFPCA1, MFPCA2)
    # get_aligned_pid_matches(MFPCA1, MFPCA2)
    # plot_polar_distribution(per_interaction_data, MFPCA1, save_path = None)
    # plot_heatmap_triplet(per_interaction_data, var1='Safe', var2='Trust', var3='Comfort', z_col='Expect', save_path=os.path.join(base_dir, 'images', 'Expect_heatmap.png'))
    
    # plot_trust_transition(per_interaction_data, save_path=os.path.join(base_dir, 'images', 'transition_matrices.png'))
    # plot_spatiotemporal_trajectories(study_data, save_path=os.path.join(base_dir, 'images', 'frechet_heatmap.png'))


    # Apply classification
    # trust_summary = compute_trust_patterns(per_interaction_data)
    # trust_labeled = classify_trust_patterns(trust_summary, n_clusters = 2)

if __name__ == '__main__':
    main()