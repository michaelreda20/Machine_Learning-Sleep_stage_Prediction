import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import glob
#from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier 
# Add new imports
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier, StackingClassifier
from imblearn.combine import SMOTEENN 
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# === Settings ===
#data_dir = r'/home/freddy/UNI/Machine/Project/Data'
data_dir = r'/home/freddy/UNI/Machine/Project/Data'

# Extract subject IDs from filenames
heart_rate_dir = os.path.join(data_dir, 'heart_rate')
heart_rate_files = glob.glob(os.path.join(heart_rate_dir, '*_heartrate.txt'))
subjects = [os.path.basename(f).split('_')[0] for f in heart_rate_files]

# === Prepare data ===
X = []
y = []

for subject in subjects:
    try:
        # print(f"Processing subject: {subject}")

        # Load files
        hr_file = os.path.join(data_dir, 'heart_rate', f"{subject}_heartrate.txt")
        accel_file = os.path.join(data_dir, 'motion', f"{subject}_acceleration.txt")
        steps_file = os.path.join(data_dir, 'steps', f"{subject}_steps.txt")
        label_file = os.path.join(data_dir, 'labels', f"{subject}_labeled_sleep.txt")

        hr_data = pd.read_csv(hr_file, sep=',', header=None, names=['time', 'heart_rate'])
        accel_data = pd.read_csv(accel_file, sep=' ', header=None, names=['time', 'x', 'y', 'z'])
        steps_data = pd.read_csv(steps_file, sep=',', header=None, names=['time', 'steps'])
        label_data = pd.read_csv(label_file, sep=' ', header=None, names=['time', 'label'])

        # Convert time columns
        hr_data['time'] = hr_data['time'].astype(float)
        accel_data['time'] = accel_data['time'].astype(float)
        steps_data['time'] = steps_data['time'].astype(float)
        label_data['time'] = label_data['time'].astype(float)

        # Merge on nearest time
        merged = pd.merge_asof(label_data.sort_values('time'), 
                               hr_data.sort_values('time'), on='time', direction='nearest')
        merged = pd.merge_asof(merged.sort_values('time'),
                               accel_data.sort_values('time'), on='time', direction='nearest')
        merged = pd.merge_asof(merged.sort_values('time'),
                               steps_data.sort_values('time'), on='time', direction='nearest')

        # Drop missing
        merged.dropna(inplace=True)

        # Smooth heart rate
        merged['heart_rate'] = merged['heart_rate'].rolling(window=5, min_periods=1).mean()

        # Acceleration magnitude
        merged['accel_mag'] = np.sqrt(merged['x']**2 + merged['y']**2 + merged['z']**2)

        # Time-based rolling features
        merged['hr_roll_mean'] = merged['heart_rate'].rolling(window=10, min_periods=1).mean()
        merged['accel_roll_std'] = merged['accel_mag'].rolling(window=10, min_periods=1).std()

        # Remove outliers
        merged = merged[(merged['heart_rate'] >= 30) & (merged['heart_rate'] <= 220)]
        merged = merged[(merged['steps'] < 1000) & (merged['accel_mag'] < 50)]

        # Valid labels only
        merged = merged[merged['label'].isin([0, 1, 2, 3, 4, 5])]

        # Merge Stage 4 into Stage 3
        merged['label'] = merged['label'].replace({4: 3})

        merged.dropna(inplace=True)

        # Features and labels
        features = merged[['heart_rate', 'steps', 'x', 'y', 'z', 'accel_mag', 'hr_roll_mean', 'accel_roll_std']].values
        labels = merged['label'].values

        X.append(features)
        y.append(labels)

    except Exception as e:
        print(f"Error processing subject {subject}: {e}")

# Concatenate
X = np.vstack(X)
y = np.hstack(y)

# # Normalize features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle imbalance
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# smote_enn = SMOTEENN(random_state=42)
# X_train, y_train = smote_enn.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# --- Linear Regression (Commented Out) ---
# model = LinearRegression()
# model.fit(X_train, y_train)

# --- XGBoost (primary model) ---
#model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
#model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)




classifiers = {
    
   "Random Forest": RandomForestClassifier(n_estimators=100, bootstrap=False, class_weight='balanced', random_state=42),

    "Logistic Regression": LogisticRegression(max_iter=100, class_weight='balanced', random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=2, metric='manhattan', n_jobs=-1, weights='distance'),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=500, random_state=42), 
    #"Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(150, 100), max_iter=500, random_state=42)71.68%70.43%
    #"Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(300, 200), max_iter=500, random_state=42) 74.20%
    #"Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(1000, 1000), max_iter=500, random_state=42)
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "Linear SVM": SVC(kernel='linear', class_weight='balanced', random_state=42),
    "Kernel SVM (RBF)": SVC(kernel='rbf', class_weight='balanced', random_state=42),
    "Naive Bayes": GaussianNB()
}




# Store accuracies
results = {}

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2,3,5], yticklabels=[0,1,2,3,5])
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    # Save confusion matrix
    cm_filename = f'confusion_matrix_{name.replace(" ", "_").lower()}.png'
    plt.savefig(cm_filename)
    plt.show()

    # Feature importance for tree models
    # if hasattr(clf, 'feature_importances_'):
    #     importances = clf.feature_importances_
    #     feature_names = ['heart_rate', 'steps', 'x', 'y', 'z', 'accel_mag', 'hr_roll_mean', 'accel_roll_std']
    #     plt.figure(figsize=(6,4))
    #     sns.barplot(x=importances, y=feature_names)
    #     plt.title(f'{name} - Feature Importance')
    #     plt.tight_layout()
    #     plt.show()

# Print summary
print("\n=== Accuracy Summary ===")
for name, acc in results.items():
    print(f"{name}: {acc*100:.2f}%")