import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier 

# === Settings ===
data_dir = '/home/freddy/UNI/Machine/Project/Data'

# === Load Subject Data ===
heart_rate_dir = os.path.join(data_dir, 'heart_rate')
heart_rate_files = glob.glob(os.path.join(heart_rate_dir, '*_heartrate.txt'))
subjects = [os.path.basename(f).split('_')[0] for f in heart_rate_files]

X_all = []
y_all = []

for subject in subjects:
    try:
        hr_file = os.path.join(data_dir, 'heart_rate', f"{subject}_heartrate.txt")
        accel_file = os.path.join(data_dir, 'motion', f"{subject}_acceleration.txt")
        steps_file = os.path.join(data_dir, 'steps', f"{subject}_steps.txt")
        label_file = os.path.join(data_dir, 'labels', f"{subject}_labeled_sleep.txt")

        hr_data = pd.read_csv(hr_file, sep=',', header=None, names=['time', 'heart_rate'])
        accel_data = pd.read_csv(accel_file, sep=' ', header=None, names=['time', 'x', 'y', 'z'])
        steps_data = pd.read_csv(steps_file, sep=',', header=None, names=['time', 'steps'])
        label_data = pd.read_csv(label_file, sep=' ', header=None, names=['time', 'label'])

        for df in [hr_data, accel_data, steps_data, label_data]:
            df['time'] = df['time'].astype(float)

        merged = pd.merge_asof(label_data.sort_values('time'), hr_data.sort_values('time'), on='time', direction='nearest')
        merged = pd.merge_asof(merged.sort_values('time'), accel_data.sort_values('time'), on='time', direction='nearest')
        merged = pd.merge_asof(merged.sort_values('time'), steps_data.sort_values('time'), on='time', direction='nearest')

        merged.dropna(inplace=True)
        merged['heart_rate'] = merged['heart_rate'].rolling(window=5, min_periods=1).mean()
        merged['accel_mag'] = np.sqrt(merged['x']**2 + merged['y']**2 + merged['z']**2)
        merged = merged[(merged['heart_rate'] >= 30) & (merged['heart_rate'] <= 220)]
        merged = merged[(merged['steps'] < 1000) & (merged['accel_mag'] < 50)]
        merged = merged[merged['label'].isin([0, 1, 2, 3, 4, 5])]
        merged['label'] = merged['label'].replace({4: 3})
        merged.dropna(inplace=True)

        X_all.append(merged[['heart_rate', 'steps', 'accel_mag']].values)
        y_all.append(merged['label'].values)

    except Exception as e:
        print(f"Error processing subject {subject}: {e}")

# === Final data preparation ===
X_all = np.vstack(X_all)
y_all = np.hstack(y_all)

feature_map = {
    "Heart Rate": 0,
    "Steps": 1,
    "Acceleration": 2
}

classifiers = {
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "SVM (Linear)": SVC(kernel='linear', class_weight='balanced', random_state=42),
    "SVM (RBF)": SVC(kernel='rbf', class_weight='balanced', random_state=42),
    "Naive Bayes": GaussianNB(),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)

}

print("\n=== MODEL PERFORMANCE BY SINGLE FEATURE ===")
for feature_name, idx in feature_map.items():
    print(f"\n--- Using Feature: {feature_name} ---")

    # Use only one feature
    X_feature = X_all[:, idx].reshape(-1, 1)

    # Normalize
    scaler = StandardScaler()
    X_feature = scaler.fit_transform(X_feature)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_feature, y_all, test_size=0.2, random_state=42)

    # Balance classes
    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    for name, clf in classifiers.items():
        print(f"\nTraining {name} with {feature_name}...")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {acc*100:.2f}%")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2,3,5], yticklabels=[0,1,2,3,5])
        plt.title(f'{name} - {feature_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.tight_layout()
        plt.show()
