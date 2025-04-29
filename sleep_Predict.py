import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import glob

# === Settings ===
data_dir = r'C:\Users\Michael Reda\Documents\Windows_Michael\one_drive_auc\OneDrive - aucegypt.edu\Documents\AUC\Spring 2025\Machine Learning\Project\Sleep_Stage_Prediction\Machine_Learning-Sleep_stage_Prediction'

# Extract subject IDs from filenames
heart_rate_dir = os.path.join(data_dir, 'heart_rate')
heart_rate_files = glob.glob(os.path.join(heart_rate_dir, '*_heartrate.txt'))

# Get subject IDs
subjects = [os.path.basename(f).split('_')[0] for f in heart_rate_files]

# === Prepare data ===
X = []  # Features: heart rate, steps, acceleration x, y, z
y = []  # Target: sleep stage

for subject in subjects:
    try:
        print(f"Processing subject: {subject}")  # Debugging: Show subject being processed

        # Load files
        hr_file = os.path.join(data_dir, rf"heart_rate\{subject}_heartrate.txt")
        accel_file = os.path.join(data_dir, rf"motion\{subject}_acceleration.txt")
        steps_file = os.path.join(data_dir, rf"steps\{subject}_steps.txt")
        label_file = os.path.join(data_dir, rf"labels\{subject}_labeled_sleep.txt")

        hr_data = pd.read_csv(hr_file, sep=',', header=None, names=['time', 'heart_rate'])
        accel_data = pd.read_csv(accel_file, sep=' ', header=None, names=['time', 'x', 'y', 'z'])
        steps_data = pd.read_csv(steps_file, sep=',', header=None, names=['time', 'steps'])
        label_data = pd.read_csv(label_file, sep=' ', header=None, names=['time', 'label'])

        print(f"Successfully loaded data for subject: {subject}")  # Debugging

        # Convert all 'time' columns to float for consistent merging
        hr_data['time'] = hr_data['time'].astype(float)
        accel_data['time'] = accel_data['time'].astype(float)
        steps_data['time'] = steps_data['time'].astype(float)
        label_data['time'] = label_data['time'].astype(float)

        # Merge all data on time
        merged = pd.merge_asof(label_data.sort_values('time'), 
                               hr_data.sort_values('time'), on='time', direction='nearest')
        merged = pd.merge_asof(merged.sort_values('time'),
                               accel_data.sort_values('time'), on='time', direction='nearest')
        merged = pd.merge_asof(merged.sort_values('time'),
                               steps_data.sort_values('time'), on='time', direction='nearest')

        # Drop rows with missing values after merging
        merged.dropna(inplace=True)

        # Add features and labels to list
        features = merged[['heart_rate', 'steps', 'x', 'y', 'z']].values
        labels = merged['label'].values

        X.append(features)
        y.append(labels)

    except Exception as e:
        print(f"Error processing subject {subject}: {e}")

# Check if any data was collected
if len(X) == 0 or len(y) == 0:
    raise ValueError("No valid data found for any subject. Please check your files and preprocessing.")

# Concatenate all subjects' data
X = np.vstack(X)
y = np.hstack(y)

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Model ===

# --- Linear Regression (Commented Out) ---
# model = LinearRegression()
# model.fit(X_train, y_train)

# --- Random Forest ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Predict ===
y_pred = model.predict(X_test)

# Make sure predictions are valid stages
valid_stages = [0, 1, 2, 3, 5]
y_pred = np.clip(y_pred, 0, 5)
y_pred = np.array([p if p in valid_stages else 0 for p in y_pred])

# === Accuracy ===
acc = accuracy_score(y_test, y_pred)

print(f"Accuracy: {acc*100:.2f}%")
