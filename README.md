# Sleep Stage Prediction from Wearable Sensor Data

## Overview

This project aims to predict sleep stages using non-invasive data collected from wearable sensors. By analyzing physiological signals like **heart rate**, **step rate**, and **acceleration**, the project demonstrates the feasibility of building a robust machine learning model to classify sleep stages, providing a practical alternative to traditional, clinical polysomnography (PSG).

## Main Features

- Data preprocessing with synchronization, cleaning, and feature engineering.
- Evaluation of multiple models across different combinations of features.
- Final model trained on all features achieves **~86% accuracy** with **Random Forest**.
- Non-invasive and suitable for real-world wearable applications.

## Dataset

The data used comes from the [Sleep-Accelerometry Dataset](https://physionet.org/content/sleep-accel/1.0.0/), which includes synchronized heart rate, accelerometry, step count, and sleep stage labels.



## Running the Project

To execute the complete pipeline using all features:

```bash
python sleep_Predict.py
