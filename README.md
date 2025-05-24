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
  python3 sleep_Predict.py



