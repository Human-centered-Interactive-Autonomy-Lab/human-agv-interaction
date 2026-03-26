import pandas as pd
import os
import logging
from build_subjective_features import *

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

    subjective_clustering_data = make_subjective_clustering_data(per_interaction_data, save_path=os.path.join(processed_data_dir, "subjective_clustering_data.csv"))
    

if __name__ == '__main__':
    main()