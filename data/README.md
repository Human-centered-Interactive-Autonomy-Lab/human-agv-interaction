# Data

This folder contains all datasets used in the human–AGV interaction study, including both raw data and processed data used for behavioral and subjective clustering and analysis.

---

## Folder Structure

### raw/
This folder contains the original raw data collected during the study.  
These datasets are used to construct all processed data.

#### Files:

- **per_second_data.csv**  
  Time-series data at approximately 1 Hz resolution.  
  This is the primary dataset used to construct behavioral features such as user speed, gaze behavior, distance, and Fréchet distance.

- **per_interaction_data.csv**  
  Aggregated data at the interaction level.  
  This dataset is primarily used to construct subjective features and includes post-survey responses.

- **eye_target_data.csv**  
  Contains eye-tracking or gaze target information during interactions.

- **pre_survey.csv**  
  Contains participants’ responses to pre-experiment surveys.

---

### processed/
This folder contains processed datasets used for clustering and analysis.

#### Behavioral Clustering Files:

- **behavioral_clustering_data.csv**  
  Dataset containing extracted behavioral features used for clustering.

- **behavioral_clustering_survey_results.csv**  
  Survey-related results aligned with behavioral clustering.

- **behavioral_kmeans_result.rds**  
  R object containing k-means clustering results for behavioral data.

- **cluster_assignments_behavioral_kmeans.csv**  
  Cluster labels assigned to each participant based on behavioral k-means clustering.

---

#### Subjective Clustering Files:

- **subjective_clustering_data.csv**  
  Dataset containing subjective features (e.g., trust, comfort, safety) used for clustering.

- **subjective_clustering_survey_results.csv**  
  Survey-related results aligned with subjective clustering.

- **subjective_kmeans_result.rds**  
  R object containing k-means clustering results for subjective data.

- **subjective_hierarchical_result.rds**  
  R object containing hierarchical clustering results for subjective data.

- **cluster_assignments_subjective_kmeans.csv**  
  Cluster labels assigned using k-means clustering on subjective data.

- **cluster_assignments_subjective_hierarchical.csv**  
  Cluster labels assigned using hierarchical clustering on subjective data.

---

## Notes

- Raw data are used to generate processed datasets through preprocessing pipelines.
- Behavioral features are primarily derived from **per_second_data.csv**.
- Subjective features and post-survey results are primarily derived from **per_interaction_data.csv**.
- Processed datasets are used for clustering, statistical analysis, and visualization.

---

## Contact

For any questions, please contact:  
mobinaa@iastate.edu