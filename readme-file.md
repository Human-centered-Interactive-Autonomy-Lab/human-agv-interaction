# Human-AGV Interaction Analysis

This project analyzes interactions between humans and Automated Guided Vehicles (AGVs) in a shared environment. It includes data processing, analysis, and modeling of various aspects of these interactions.

## Project Structure

```
human-agv-interaction/
│
├── data/
│   └── Per_Interaction_Data_Merged.csv
│
├── notebooks/
│   └── per-interaction-data-analysis.ipynb
│
├── src/
│   └── data_processing.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/human-agv-interaction.git
   cd human-agv-interaction
   ```

2. Create and activate a conda environment:
   ```
   conda create -n ros_env python=3.11
   conda activate ros_env
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Data

The main dataset is `Per_Interaction_Data_Merged.csv`, located in the `data/` directory. This file contains information about individual human-AGV interactions, including spatial and temporal data.

To load the data:

```python
import pandas as pd
import os

box_path = os.path.expanduser("~/Library/CloudStorage/Box-Box/Human_AGV_project/ISU_Modeling/Code")
file_path = os.path.join(box_path, "Per_Interaction_Data_Merged.csv")
df = pd.read_csv(file_path)
```

## Analysis

The main analysis is performed in the Jupyter notebook `per-interaction-data-analysis.ipynb`. This notebook includes:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Statistical analysis
- Modeling (including XGBoost)

## Contributing

1. Create a new branch for your feature:
   ```
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```
   git add .
   git commit -m "Description of your changes"
   ```

3. Push your changes to the remote repository:
   ```
   git push origin feature/your-feature-name
   ```

4. Create a pull request on GitHub for review.

## Troubleshooting

For common issues and their solutions, please refer to the `project_guide.md` file in the repository.

## Contact

For any questions or concerns, please contact [Your Name] at [your.email@example.com].
