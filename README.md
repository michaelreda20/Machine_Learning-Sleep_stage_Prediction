# Machine_Learning-Sleep_stage_Prediction

# Sleep Stage Prediction Project

## Project Description
This project aims to predict sleep stages (light, deep, REM) using non-invasive wearable sensor data, including heart rate, acceleration, and step count. By leveraging machine learning models, the system provides a simpler alternative to clinical polysomnography (PSG). The dataset used is the **Sleep-Accelerometry Dataset**, and the workflow includes data preprocessing, feature engineering, and model evaluation.

## Key Features
- **Data Synchronization**: Temporal alignment of multi-sensor data.
- **Feature Engineering**: Derived features like acceleration magnitude and rolling statistics.
- **Imbalance Correction**: SMOTE for addressing class imbalance.
- **Model Comparison**: Evaluated Random Forest, KNN, CatBoost, SVM, and others.
- **High Accuracy**: Achieved **85.95% accuracy** (Random Forest) using all features.

## Installation
### Prerequisites
- Python 3.7+
- Libraries:  
  ```bash
  pip install scikit-learn imbalanced-learn pandas numpy catboost

# Single-feature experiments
python experiment_hr_only.py          # Heart rate only
python experiment_steps_only.py       # Steps only
python experiment_acceleration_only.py  # Acceleration only

# Two-feature combinations
python experiment_acceleration_steps.py     # Acceleration + steps
python experiment_acceleration_hr.py        # Acceleration + heart rate
python experiment_steps_hr.py               # Steps + heart rate

Results
Best Performances
All Features Combined:

Random Forest: 85.95%

K-Nearest Neighbors (KNN): 84.05%

CatBoost: 77.04%

Single-Feature Results
Heart Rate (KNN): 42.76%

Acceleration (Linear SVM): 32.47%

Steps (KNN): 21.53%

Two-Feature Combinations
Steps + Heart Rate (KNN): 45.10%

Acceleration + Heart Rate (SVM/Random Forest): ~34%

Contributing
Contributions are welcome! Open an issue or submit a pull request for improvements.

License
MIT License.

Contact
Michael Reda: 900203291

Freddy Amgad: 900203088

