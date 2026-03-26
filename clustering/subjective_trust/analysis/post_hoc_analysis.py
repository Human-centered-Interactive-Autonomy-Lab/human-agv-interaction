import pandas as pd
from pathlib import Path
import os
from post_hoc_analysis_utils import *

def main():
    processed_data_dir = os.path.join(os.path.join(os.getcwd(), "data"), "processed")
    processed_dir = Path(processed_data_dir)

    raw_data_dir = os.path.join(os.path.join(os.getcwd(), "data"), "raw")
    raw_dir = Path(raw_data_dir)
    
    subjective_clustering_assignments = pd.read_csv(os.path.join(processed_dir, 'cluster_assignments_subjective_kmeans.csv'))
    subjective_clustering_data = pd.read_csv(os.path.join(processed_dir, 'subjective_clustering_data.csv'))

    # Behavioral clustering process visulization
    save_df_path = os.path.join(os.path.join(os.path.join(os.getcwd(), "data"), "processed"), "subjective_clustering_survey_results.csv") 

    assumption_results = run_pre_post_tests_auto(
    df = subjective_clustering_data,
    cluster_df = subjective_clustering_assignments,
    save_path = save_df_path)
    
    print(assumption_results)

if __name__ == '__main__':
    main()