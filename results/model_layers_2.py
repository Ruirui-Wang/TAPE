import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

markersize = 16

# Load the CSV file
file_path = "/hkfs/work/workspace/scratch/cc7738-rebuttal/TAPE_chen/TAPE/results/ablation_study_layer.csv"
df = pd.read_csv(file_path)

# Extracting GCN layers (X-axis) and AUC values (Y-axis)
df["Layer"] = pd.to_numeric(df["Layer"], errors="coerce")  # Convert layer column to numeric
df["val"] = df["AUC"].astype(str).str.split("±").str[0].astype(float)  # Extract numeric part before "±"
df["std"] = df["AUC"].astype(str).str.split("±").str[1].astype(float)  # Standard deviation

# Extract data for plotting
gnn_layers = df["Layer"].values
auc_values = df["val"].values
auc_std = df["std"].values

# Create figure
plt.figure(figsize=(6, 5))
sns.set_style("whitegrid")

# Plot with error bands
plt.plot(gnn_layers, auc_values, 'o-', color='royalblue', label="Cora", markersize=markersize)
plt.fill_between(gnn_layers, auc_values - auc_std, auc_values + auc_std, 
                 color='royalblue', alpha=0.2)

# Plot for the second dataset (Pubmed)
df["val"] = df["AP"].astype(str).str.split("±").str[0].astype(float)  # Extract numeric part before "±"
df["std"] = df["AP"].astype(str).str.split("±").str[1].astype(float)  # Standard deviation

# Extract data for plotting
auc_values = df["val"].values
auc_std = df["std"].values

plt.plot(gnn_layers, auc_values, 'o-', color='mediumseagreen', label="Pubmed", markersize=markersize)
plt.fill_between(gnn_layers, auc_values - auc_std, auc_values + auc_std, 
                 color='mediumseagreen', alpha=0.2)

fontsize = 24
# Labels and title
plt.xlabel("Number of GCN Layers", fontsize=fontsize)
plt.ylabel("Test Accuracy", fontsize=fontsize)
plt.title("Effect of GCN Depth on AUC", fontsize=fontsize)
plt.legend(loc="lower left", fontsize=fontsize)
plt.xticks(fontsize=fontsize)  # Change x-axis tick label size
plt.yticks(fontsize=fontsize)  # Change y-axis tick label size

# Alternative:
plt.tick_params(axis='both', labelsize=fontsize)  # Change both x and y axis tick label sizes


# Formatting
plt.xticks(gnn_layers)
plt.yticks(np.arange(70, 102, 10))
plt.ylim(70, 105)
plt.tight_layout()

# Save figure
root = '/hkfs/work/workspace/scratch/cc7738-rebuttal/TAPE_chen/TAPE/results'
plt.savefig(f"{root}/_layers.png", dpi=300)
