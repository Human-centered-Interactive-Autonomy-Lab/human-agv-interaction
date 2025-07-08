import pandas as pd
import os
import random
from random import choice
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from visualization import *
from visualization.transition_matrix import plot_trust_transition
from visualization.plot_frechet_heatmap import plot_spatiotemporal_trajectories
from visualization.radial_AGV_trust_map import plot_agv_direction_trust
from visualization.plot_3d import plot_3d_response_surface
from visualization.dependent_differences import plot_dependent_differences_by_drate
from visualization.feature_trends import plot_feature_trends_by_rating
from visualization.time_series_trends import visualize
from visualization.polar_distribution import polar_distribution
from analysis.stat_test import analyze_cross_first_difference
from analysis.rating_prediction_model import train_regression_models
from clustering.preprocessing import DataProcessor
from clustering.processing import AGVFeatureTimeseries
from clustering.timeseries_clustering import time_series_clustering

def main():
    # Load Data
    base_dir = os.getcwd()
    print(base_dir)
    data_path = os.path.join(base_dir, 'data')
    
    per_interaction_data = pd.read_csv(os.path.join(data_path, 'per_interaction_data.csv'))
    study_data = pd.read_csv(os.path.join(data_path, 'study_data_processed.csv'))
    processor = DataProcessor(study_data).run_mappings()

    

    print(per_interaction_data.head())
    print(per_interaction_data.columns.tolist())
    # print(study_data.columns.tolist())
    # plot_trust_heatmap(per_interaction_data, var1='Safe', var2='Expect', var3='Comfort', z_col='Trust', save_path=os.path.join(base_dir, 'images'))
    # plot_trust_transition(per_interaction_data, save_path=os.path.join(base_dir, 'images', 'transition_matrices.png'))
    # plot_spatiotemporal_trajectories(study_data, save_path=os.path.join(base_dir, 'images', 'frechet_heatmap.png'))
    # plot_agv_direction_trust(per_interaction_data, trust_col='Safe', direction_col='AGV_Approaching', save_path=os.path.join(base_dir, 'images', 'agv_trust_radial.png'))
    # plot_3d_response_surface(per_interaction_data, x_col='Trust', y_col='Safe', z_col='Expect', title= "Expect Surface", save_path=os.path.join(base_dir, 'images'))
    # plot_dependent_differences_by_drate(per_interaction_data.sort_values(by=['PID', 'StartTime'], ascending=[True, True]))
    # print(analyze_cross_first_difference(per_interaction_data))
    # plot_feature_trends_by_rating(per_interaction_data, save_path=os.path.join(base_dir, 'images', 'feature_trends.png'))
    # train_regression_models(per_interaction_data)
    agv_features = ['AGV_X', 'AGV_Y', 'AGV_Z', 'AGV_Pitch', 'AGV_Yaw', 'AGV_Roll', 'AGV_spd']
    '''
    analyzer = AGVFeatureTimeseries(study_data, agv_features) \
    .scale_features() \
    .compute_statistics() \
    .smooth_result()
    '''
    polar_distribution(per_interaction_data, save_path=os.path.join(base_dir, 'images', 'polar_distribution_of_dvs.png'))

    
    # result = analyzer.get_result()
    # visualize(result, save_path=os.path.join(base_dir, 'images', 'feature_trends.png'))

    # print(time_series_clustering(result, per_interaction_data, agv_features))





if __name__ == '__main__':
    main()
