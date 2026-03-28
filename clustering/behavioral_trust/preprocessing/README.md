# Behavioral Data Preprocessing

This folder contains scripts for preprocessing raw data and extracting behavioral features used in clustering and analysis.

## File Descriptions

### behavioral_features_diagnostics.py
This script contains visualization functions used to analyze behavioral features.  
It includes functions for generating correlation matrices.  
These functions can be run independently for exploratory analysis.

### build_behavioral_features.py
This script is responsible for extracting all behavioral features from the processed data.  
It constructs the feature set used for clustering and downstream analysis.

### prepare_behavioral_clustering_data.py
This file contains helper (elementary) functions that support the feature-building process.  
It includes utility functions for cleaning, transforming, and organizing the data.

### run_python_prep_for_behavioral_clustering.py
This is the main preprocessing script.  
It takes the 1 Hz data and generates the full behavioral feature dataset used for clustering.  
Running this script will produce the dataset that is used throughout the analysis pipeline.

## Notes
- The preprocessing pipeline converts raw interaction data into structured behavioral features.
- The final output of this pipeline is the dataset used for clustering.
- Diagnostic tools are available to validate feature relationships before clustering.

## Contact
For any questions, please contact:  
mobinaa@iastate.edu