import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv("model_accuracy_results.csv")

# Add a column to indicate number of parameters used
df["Num Parameters"] = df["Features"].apply(lambda x: x.count('+') + 1 if '+' in x else (3 if 'All' in x else 1))

# Plot 1: Accuracy vs Number of Parameters for all models (average per parameter count)
plt.figure(figsize=(10, 6))
sns.barplot(data=df, x="Num Parameters", y="Accuracy", ci=None)
plt.title("Average Accuracy vs Number of Parameters")
plt.ylabel("Accuracy (%)")
plt.xlabel("Number of Parameters")
plt.tight_layout()
plt.savefig("accuracy_vs_num_parameters.png")  # Save the figure
plt.show()

# Plot 2: Accuracy of all models using the best-performing 3 parameters (All features)
df_3params = df[df["Num Parameters"] == 3]

plt.figure(figsize=(12, 6))
sns.barplot(data=df_3params, x="Model", y="Accuracy", palette="viridis")
plt.title("Model Accuracy Using All Three Parameters")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("accuracy_all_three_parameters.png")  # Save the figure
plt.show()
