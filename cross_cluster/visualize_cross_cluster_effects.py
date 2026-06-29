import pandas as pd
from pathlib import Path
import os

from analysis_utils import *


def read_csv_with_source(path):
    dataframe = pd.read_csv(path)
    dataframe.attrs['source_name'] = Path(path).name
    return dataframe

def main():
    processed_data_dir = os.path.join(os.path.join(os.getcwd(), "data"), "processed")
    processed_dir = Path(processed_data_dir)

    raw_data_dir = os.path.join(os.path.join(os.getcwd(), "data"), "raw")
    raw_dir = Path(raw_data_dir)
    
    subjective_clustering_assignments = read_csv_with_source(os.path.join(processed_dir, 'cluster_assignments_subjective_kmeans.csv'))
    behavioral_clustering_assignments = read_csv_with_source(os.path.join(processed_dir, 'cluster_assignments_behavioral_kmeans.csv'))
    subjective_clustering_data = read_csv_with_source(os.path.join(processed_dir, 'subjective_clustering_data.csv'))
    behavioral_clustering_data = read_csv_with_source(os.path.join(processed_dir, 'behavioral_clustering_data.csv'))
    per_second_data = read_csv_with_source(os.path.join(raw_dir, 'per_second_data.csv'))

    agreement_metrics, contingency_table = analyze_cluster_agreement(
        behavioral_clustering_assignments,
        subjective_clustering_assignments,
        save_path=processed_dir / "cluster_agreement_contingency_table.csv",
        metrics_save_path=processed_dir / "cluster_agreement_metrics.csv",
    )
    print("Cluster agreement metrics:")
    for metric, value in agreement_metrics.items():
        print(f"  {metric}: {value:.3f}" if isinstance(value, float) else f"  {metric}: {value}")
    print("\nContingency table:")
    print(contingency_table)

    speed_summary, _, speed_test, speed_unit_audit = analyze_initial_walking_speed_by_cluster(
        per_second_data,
        behavioral_clustering_assignments,
        cluster_labels={1: "Skeptical", 2: "Deliberate"},
        report_save_path=processed_dir / "all_32_trials_until_11_seconds_before_interaction_cluster_report.csv",
    )
    plot_pre_interaction_walking_speed_trends(
        speed_summary,
        save_path=Path.cwd() / "cross_cluster" / "figures" / "all_32_trials_11_second_buffer_walking_speed_trends.png",
    )
    print("\nInitial walking speed by behavioral cluster:")
    print(speed_summary.to_string(index=False))
    print(f"Unit audit: {speed_unit_audit}")
    print(f"Statistical test: {speed_test}")

    # Behavioral clustering process visulization
    columns_to_visualize = ['Gaze_Angle_to_AGV', 'User_Speed', 'Gaze_Instability', 'Frechet_Distance']
    save_fig_path=Path.cwd() / "cross_cluster" / "figures" 
    # plot_agv_user_fov(data=per_second_data, save_path=os.path.join(save_fig_path, 'AGV_in_User_Field_of_View_Visualization.png'), n_samples=2, fov_half_angle_deg=45, scale_factor=100.0, gaze_arrow_len_m=120.0, random_state=47)
    # plot_cluster_agreement_heatmap(behavioral_clustering_assignments, subjective_clustering_assignments)
    # plot_cluster_split_stacked_bars(behavioral_clustering_assignments, subjective_clustering_assignments)
    # plot_agreement_group_counts(behavioral_clustering_assignments, subjective_clustering_assignments)
    # plot_cluster_agreement_sankey(behavioral_clustering_assignments, subjective_clustering_assignments, save_path=os.path.join(save_fig_path, 'cluster_agreement_sankey.png'))
    # Plot efficiency (trial time) comparison across clustering methods
    plot_efficiency_cross_cluster(data=subjective_clustering_data, behavioral_cluster_file=behavioral_clustering_assignments, subjective_cluster_file=subjective_clustering_assignments,
    save_path=os.path.join(save_fig_path, 'efficiency_cross_cluster_comparison.png')
    )

    plot_cross_first_cross_cluster(
        df=subjective_clustering_data,
        behavioral_cluster_file=behavioral_clustering_assignments,
        subjective_cluster_file=subjective_clustering_assignments,
        save_path=os.path.join(save_fig_path, 'cross_first_cross_cluster_comparison.png')
    )

if __name__ == '__main__':
    main()
