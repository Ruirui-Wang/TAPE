import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
plt.figure(figsize=(6,5))
sns.set_style("whitegrid")

# Plot with error bands
plt.plot(gnn_layers, auc_values, 'o-', color='royalblue', label="Cora", markersize=6)
plt.fill_between(gnn_layers, auc_values - auc_std, auc_values + auc_std, 
                 color='royalblue', alpha=0.2)

df["Layer"] = pd.to_numeric(df["Layer"], errors="coerce")  # Convert layer column to numeric
df["val"] = df["AP"].astype(str).str.split("±").str[0].astype(float)  # Extract numeric part before "±"
df["std"] = df["AP"].astype(str).str.split("±").str[1].astype(float)  # Standard deviation

# Extract data for plotting
gnn_layers = df["Layer"].values
auc_values = df["val"].values
auc_std = df["std"].values

plt.plot(gnn_layers, auc_values, 'o-', color='mediumseagreen', label="Pubmed", markersize=6)
plt.fill_between(gnn_layers, auc_values - auc_std, auc_values + auc_std, 
                 color='mediumseagreen', alpha=0.2)

# Labels and title
plt.xlabel("Number of GCN Layers", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.title("Effect of Graph Encoder Depth on Test AUC", fontsize=13)
# Save figure
root = '/hkfs/work/workspace/scratch/cc7738-rebuttal/TAPE_chen/TAPE/results'
plt.savefig(f"{root}/ablation_study.png", dpi=300)

# Formatting
plt.xticks(gnn_layers)
plt.yticks(np.arange(70, 102, 10))
plt.ylim(70, 105)
plt.legend(loc="upper left", fontsize=11)
plt.tight_layout()

# Save figure
root = '/hkfs/work/workspace/scratch/cc7738-rebuttal/TAPE_chen/TAPE/results'
plt.savefig(f"{root}/ablation_study.png", dpi=300)

