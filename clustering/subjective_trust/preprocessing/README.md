# Subjective Data Preprocessing

This folder contains scripts for preprocessing subjective (self-reported) data and preparing it for clustering and analysis.

## File Descriptions

### subjective_features_diagnostics.py
This script contains visualization functions used to analyze subjective features.  
It includes functions for generating correlation matrices.  
These functions can be run independently for exploratory analysis.

### build_subjective_features.py
This script is responsible for constructing subjective features from the dataset.  
It prepares variables such as trust, comfort, and safety measures for clustering.

### run_python_prep_for_subjective_clustering.py
This is the main preprocessing script.  
It processes the raw subjective data and generates the final dataset used for clustering.  
Running this script will produce the dataset used throughout the subjective analysis pipeline.

## Notes
- The preprocessing pipeline converts raw subjective responses into structured features.
- The final output of this pipeline is the dataset used for clustering.
- Diagnostic tools are available to validate relationships between subjective measures before clustering.

## Contact
For any questions, please contact:  
mobinaa@iastate.edu