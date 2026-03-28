# Clustering Derivation and Analysis

This repository contains the clustering derivation, analysis, and supporting materials for the paper:

**"Measuring and Understanding Trust in Motion: Behavioral Archetypes in Human-Automated Guided Vehicle Interactions"**

---

## Overview

This project focuses on identifying and analyzing trust-related behavioral and subjective patterns in human–AGV interactions.  
It includes data preprocessing, feature extraction, clustering, post-hoc statistical analysis, and cross-cluster comparisons.

Each subfolder in this repository contains its own `README.md` file with detailed explanations of its structure and contents.

---

## Folder Structure

### clustering/
This folder contains all scripts and outputs related to clustering.  
It includes:
- Behavioral clustering (based on extracted behavioral features)  
- Subjective clustering (based on self-reported measures)  
- Clustering methods such as k-means and hierarchical clustering  
- Evaluation metrics (e.g., silhouette scores, elbow method)

---

### cross_cluster/
This folder contains analysis and visualizations comparing behavioral and subjective clusters.  
It includes:
- Cross-cluster comparisons (e.g., agreement, overlap)  
- Efficiency and interaction-based comparisons  
- Visualizations such as Sankey diagrams and heatmaps  
- Analysis of metrics like Fréchet distance and cross-first behavior  

---

### data/
This folder contains all datasets used in the study.  
It includes:
- Raw data collected during the experiment  
- Processed datasets used for feature extraction and clustering  
- Cluster assignments and clustering outputs  

---

## Notes

- Behavioral clustering identifies patterns such as **Skeptical** and **Deliberate** user behaviors.
- Subjective clustering identifies groups with **higher trust** and **lower trust** based on self-reported measures.
- Cross-cluster analysis helps understand the relationship between observed behavior and self-reported trust.

---

## Contact

For any questions, please contact:  
mobinaa@iastate.edu