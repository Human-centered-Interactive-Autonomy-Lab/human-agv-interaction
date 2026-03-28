# Cross-Cluster Analysis

This folder contains scripts and visualizations for comparing behavioral and subjective clustering results in the human–AGV interaction study.

## File Descriptions

### visualize_cross_cluster_effects.py
This script is responsible for generating the main cross-cluster visualizations.  
It includes:
- Average interaction time comparisons  
- Cross-first (who initiates movement) comparisons  
- Additional comparative plots between clusters  

### analysis_utils.py
This file contains helper functions used in the cross-cluster analysis and visualization process.  
It provides reusable utilities for data processing and plotting.

## Figures

This folder includes several figures that illustrate relationships between behavioral and subjective clusters:

- **cluster_agreement_sankey.png**  
  Shows how participants are distributed across behavioral and subjective clusters.

- **cross_first_cross_cluster_comparison.png**  
  Compares which agent (human or AGV) moves first across clusters.

- **efficiency_cross_cluster_comparison.png**  
- **efficiency_cross_cluster_comparison_behavioral.png**  
- **efficiency_cross_cluster_comparison_subjective.png**  
  These figures illustrate efficiency comparisons across clusters.

- **AGV_in_User_Field_of_View_Visualization.png**  
  Visualizes how often the AGV appears in the user’s field of view.

- **frechet_heatmap.png**  
- **why_frechet_distance.svg**  
  These figures demonstrate the role and effectiveness of Fréchet distance in analyzing trajectories.

## Notes
- This module focuses on comparing clustering results across different perspectives (behavioral vs. subjective).
- Visualizations include:
  - Average interaction time  
  - Cross-first behavior  
  - Cluster agreement and overlap  
- Additional figures explore:
  - The efficiency of interactions  
  - The effectiveness of Fréchet distance  
  - How many data points from one clustering belong to clusters in the other  

## Contact
For any questions, please contact:  
mobinaa@iastate.edu