import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data (replace this with your actual CSV file path)
data = pd.read_csv("/hkfs/work/workspace/scratch/cc7738-2025_whole/TAPE_chen/TAPE/results/plots/plot1/complexity_cora.csv")

# Process the data
data['AUC'] = data['AUC'].str.split(' ± ').str[0].astype(float)
data['MRR'] = data['MRR'].str.split(' ± ').str[0].astype(float)
data.sort_values(by='AUC', inplace=True)
data.reset_index(drop=True, inplace=True)

methods = data['method']
inference_time = data['inference time']
AUC_val = data['AUC']
params = data['parameters']

# Normalize inference time for bubble sizes
norm_time = (inference_time - np.min(inference_time)) / (np.max(inference_time) - np.min(inference_time))

# Define categories and colors
categories = {
    "Heuristic": {"color": "gray", "marker": ">"},
    "GraphEmb": {"color": "pink", "marker": ">"},
    "GNN": {"color": "gold", "marker": ">"},
    "GNN4LP": {"color": "gold",  "marker": ">"},
    "PLM-based": {"color": "skyblue",  "marker": ">"},
    "Proposed": {"color": "green",  "marker": ">"},
}

# Assign categories (this should match your dataset's methods)

# Plot setup
plt.figure(figsize=(6, 6))
for i, row in data.iterrows():
    category = row['Category']
    color = categories[category]['color']
    marker = categories[category]['marker']
    
    # Scatter plot with bubble size
    bubble_size = 100
    plt.scatter(
        row['AUC'], np.log10(row['parameters']),
        s=bubble_size, color=color, alpha=0.7, label=row['method'] if i == 0 else "",
        marker=marker
    )
    # Add text annotations
    plt.text(
        row['AUC'], np.log10(row['parameters']), row['method'],
        fontsize=10, ha='center', va='bottom'
    )

# Labels and title
plt.xlabel("AUC (%)", fontsize=14)
plt.ylabel("Number of Params (Log Scale)", fontsize=14)
plt.title("Comparison of Different Methods", fontsize=16)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Legend
legend_elements = [
    plt.Line2D([0], [0], marker=categories[cat]['marker'], color='w',
               markerfacecolor=categories[cat]['color'], markersize=10, label=cat)
    for cat in categories
]
legend_elements.append(
    plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='black',
               markerfacecolor='white', markersize=10, label='Inference Time')
)

# Save and show the plot
plt.tight_layout()
plt.savefig("/hkfs/work/workspace/scratch/cc7738-2025_whole/TAPE_chen/TAPE/results/plots/plot1/AUC_plot_benchmark.pdf")
plt.show()
