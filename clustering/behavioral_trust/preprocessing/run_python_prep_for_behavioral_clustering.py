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

if __name__ == '__main__':
    main()