# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # Load the data (replace this with your actual CSV file path)
# data = pd.read_csv("/hkfs/work/workspace/scratch/cc7738-2025_whole/TAPE_chen/TAPE/results/plots/plot1/complexity_cora.csv")
# FONTSIZE = 20
# # Process the data
# data['AUC'] = data['AUC'].str.split(' ± ').str[0].astype(float)
# data['MRR'] = data['MRR'].str.split(' ± ').str[0].astype(float)
# data.sort_values(by='AUC', inplace=True)
# data.reset_index(drop=True, inplace=True)

# methods = data['method']
# inference_time = data['inference time']
# AUC_val = data['AUC']
# params = data['parameters']

# # Normalize inference time for bubble sizes
# norm_time = (inference_time - np.min(inference_time)) / (np.max(inference_time) - np.min(inference_time))

# # Define categories and colors
# categories = {
#     "Heuristic": {"color": "gray", "marker": ">"},
#     "GraphEmb": {"color": "pink", "marker": ">"},
#     "GNN": {"color": "gold", "marker": ">"},
#     "GNN4LP": {"color": "gold",  "marker": ">"},
#     "PLM-based": {"color": "skyblue",  "marker": ">"},
#     "Proposed": {"color": "green",  "marker": ">"},
# }

# # Assign categories (this should match your dataset's methods)

# # Plot setup
# plt.figure(figsize=(6, 6))
# for i, row in data.iterrows():
#     category = row['Category']
#     color = categories[category]['color']
#     marker = categories[category]['marker']
    
#     # Scatter plot with bubble size
#     bubble_size = 100
#     plt.scatter(
#         row['AUC'], np.log10(row['parameters']),
#         s=bubble_size, color=color, alpha=0.7, label=row['method'] if i == 0 else "",
#         marker=marker
#     )
#     # Add text annotations
#     plt.text(
#         row['AUC'], np.log10(row['parameters']), row['method'],
#         fontsize=FONTSIZE, ha='center', va='bottom'
#     )


# # Labels and title
# plt.xlabel("AUC (%)", fontsize=FONTSIZE)
# plt.ylabel("Number of Params (Log Scale)", fontsize=FONTSIZE)
# plt.title("Comparison of Different Methods", fontsize=FONTSIZE)
# plt.grid(True, which="both", linestyle="--", linewidth=0.5)
# plt.xticks(fontsize=FONTSIZE)
# plt.yticks(fontsize=FONTSIZE)
# # Legend
# legend_elements = [
#     plt.Line2D([0], [0], marker=categories[cat]['marker'], color='w',
#                markerfacecolor=categories[cat]['color'], markersize=10, label=cat)
#     for cat in categories
# ]
# legend_elements.append(
#     plt.Line2D([0], [0], marker='o', color='w', markeredgecolor='black',
#                markerfacecolor='white', markersize=10, label='Inference Time')
# )

# # Save and show the plot
# plt.tight_layout()
# plt.savefig("/hkfs/work/workspace/scratch/cc7738-2025_whole/TAPE_chen/TAPE/results/plots/plot1/AUC_plot_benchmark.pdf")
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogLocator
from adjustText import adjust_text  # pip install adjustText

# Load the data
data = pd.read_csv("/hkfs/work/workspace/scratch/cc7738-2025_whole/TAPE_chen/TAPE/results/plots/plot1/complexity_cora.csv")
FONTSIZE = 18

# Process the data
data['AUC'] = data['AUC'].str.split(' ± ').str[0].astype(float)
data['MRR'] = data['MRR'].str.split(' ± ').str[0].astype(float)
data.sort_values(by='AUC', inplace=True)
data.reset_index(drop=True, inplace=True)

# Prepare variables
methods = data['method']
inference_time = data['inference time']
AUC_val = data['AUC']
params = data['parameters']

# Normalize inference time for bubble sizes
bubble_sizes = 200 * (inference_time - inference_time.min()) / (inference_time.max() - inference_time.min()) + 50

# Define categories (with distinct markers!)
categories = {
    "Heuristic": {"color": "gray", "marker": "s"},
    "GraphEmb": {"color": "pink", "marker": "D"},
    "GNN": {"color": "gold", "marker": "o"},
    "GNN4LP": {"color": "orange", "marker": "^"},
    "PLM-based": {"color": "skyblue", "marker": "v"},
    "Proposed": {"color": "green", "marker": "P"},
}

# Plot
plt.figure(figsize=(8, 7))
texts = []
for i, row in data.iterrows():
    cat = row['Category']
    plt.scatter(
        row['AUC'], np.log10(row['parameters']),
        s=bubble_sizes[i],
        color=categories[cat]['color'],
        alpha=0.7,
        edgecolor="k",
        linewidth=0.6,
        marker=categories[cat]['marker']
    )
    # Add labels (collected for adjust_text)
    texts.append(plt.text(row['AUC'], np.log10(row['parameters']), row['method'], fontsize=12))

# Adjust labels to avoid overlap
adjust_text(texts, arrowprops=dict(arrowstyle="->", color='black', lw=0.5))

# Labels and title
plt.xlabel("AUC (%)", fontsize=FONTSIZE)
plt.ylabel("Number of Parameters (log scale)", fontsize=FONTSIZE)
plt.title("Comparison of Link Prediction Methods", fontsize=FONTSIZE)

# Better grid and ticks
plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.6)
plt.xticks(fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.gca().yaxis.set_major_locator(LogLocator(base=10.0, subs=None, numticks=5))

# Legend: Categories + Bubble size meaning
legend_elements = [
    plt.Line2D([0], [0], marker=v['marker'], color='w',
               markerfacecolor=v['color'], markeredgecolor="k",
               markersize=12, label=cat)
    for cat, v in categories.items()
]
legend_elements.append(
    plt.scatter([], [], s=100, color="lightgray", edgecolor="k", alpha=0.6, label="Shorter inference")
)
legend_elements.append(
    plt.scatter([], [], s=400, color="lightgray", edgecolor="k", alpha=0.6, label="Longer inference")
)

plt.legend(handles=legend_elements, fontsize=FONTSIZE-2, loc="upper left", frameon=True)

plt.tight_layout()
plt.savefig("/hkfs/work/workspace/scratch/cc7738-2025_whole/TAPE_chen/TAPE/results/plots/plot1/AUC_plot_benchmark_improved.pdf")
plt.show()
