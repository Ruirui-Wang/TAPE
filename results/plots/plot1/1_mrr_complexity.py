# import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
# import numpy as np

# # Set up the theme and font
# sns.set_theme(style="whitegrid", palette="pastel")
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Amiri"
# })

# def load_data(file_path):
#     data = pd.read_csv(file_path)
#     # Convert AUC and MRR to numerical values
#     data['AUC'] = data['AUC'].str.split(' ± ').str[0].astype(float)
#     data['MRR'] = data['MRR'].str.split(' ± ').str[0].astype(float)
#     return data

# # Load data
# file_path = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/results/plots/plot1/complexity_cora.csv'
# data = load_data(file_path)

# # Sort by AUC and reset index
# sorted_data = data.sort_values(by='AUC', ascending=False)  # Sort descending to get the top AUC values
# sorted_data.reset_index(drop=True, inplace=True)

# # Prepare data for plotting
# plot_data = sorted_data[['method', 'parameters', 'inference time', 'AUC', 'MRR']]
# methods = plot_data['method'].tolist()
# inference_time = plot_data['inference time'].tolist()
# MRR_val = plot_data['MRR'].tolist()
# AUC_val = plot_data['AUC'].tolist()
# params = [i/100 for i in plot_data['parameters'].tolist()]

# top_models = methods[:3]  

# top_colors = ['gold', 'orange', 'lightgreen']

# cmap = plt.colormaps['Blues']
# colors = cmap(np.linspace(0.3, 1, len(methods))[::-1])  

# # Create plot
# fig, ax = plt.subplots(facecolor='white')

# x_points = []
# y_points = []
# offset_x = 0.2
# offset_y = 0.2
# font_size = 8  

# param_threshold = 10000

# for i, (method, time, metric, color, param) in enumerate(list(zip(methods, inference_time, AUC_val, colors, params))):
#     if i % 2 == 0: 
#         ax.text(i - offset_x, metric - offset_y, f'{method}', fontsize=font_size, ha='left', va='bottom')
    
#     # Set the color for the top models
#     if method in top_models:
#         color = top_colors[top_models.index(method)]
    
#     if param < param_threshold:
#         alpha_value = 0.4  # Less transparent
#     else:
#         alpha_value = 0.1 # More transparent
    
#     ax.scatter(i, metric, s=param, color=color, alpha=alpha_value, edgecolor='b', linewidth=2)
#     x_points.append(i)
#     y_points.append(metric)

# # Adding red "x" markers
# for i, (method, time, metric, color, param) in enumerate(list(zip(methods, inference_time, AUC_val, colors, params))):
#     ax.scatter(i, metric, s=10, color='red', marker='x', label=method)

# # Customize x-ticks
# ax.set_xticks(range(len(methods)))
# ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=12)

# ax.set_ylabel('AUC (\%)', fontsize=14)
# ax.grid(True, which="both", ls="--")

# # Add vertical lines at specific positions with spacing
# vertical_lines = {
#     'PageRank': 'DeepWalk', 
#     'Node2vec': 'SAGE', 
#     'GCN': 'SEAL', 
#     'ELPH': 'Minilm', 
#     'Llama': 'FT-bert'
# }

# y_list = [[0.4, 0.6], [0.5, 0.8],  [0.7, 0.8], [0.7, 0.9], [0.9, 0.95]]
# for (start, end), y in zip(vertical_lines.items(), y_list):
#     start_idx = methods.index(start)
#     end_idx = methods.index(end)
#     mid_point = (start_idx + end_idx) / 2
#     ax.axvline(x=mid_point+0.3, color='gray', linestyle='--', linewidth=1.5,
#                ymin=y[0], ymax=y[1])  # Adjust ymin and ymax to control the vertical span

# # Tick parameters
# ax.tick_params(axis='both', which='both', direction='in', length=6)
# ax.tick_params(axis='both', which='major', labelsize=12)
# ax.tick_params(axis='both', which='minor', labelsize=8)

# # Title
# ax.set_title('Comparison of Different Methods', fontsize=16)

# # Adjust layout and save plot
# plt.tight_layout()
# plt.savefig('AUC1_plot_benchmark.pdf')

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set up the theme and font
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Amiri"
})

def load_data(file_path):
    data = pd.read_csv(file_path)
    # Convert AUC and MRR to numerical values
    data['AUC'] = data['AUC'].str.split(' ± ').str[0].astype(float)
    data['MRR'] = data['MRR'].str.split(' ± ').str[0].astype(float)
    return data

# Load data
file_path = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/results/plots/plot1/complexity_cora.csv'
data = load_data(file_path)

# Sort by AUC and reset index
sorted_data = data.sort_values(by='AUC', ascending=True)  # Sort descending to get the top AUC values
sorted_data.reset_index(drop=True, inplace=True)

# Prepare data for plotting
plot_data = sorted_data[['method', 'parameters', 'inference time', 'AUC', 'MRR']]
methods = plot_data['method'].tolist()
inference_time = plot_data['inference time'].tolist()
MRR_val = plot_data['MRR'].tolist()
AUC_val = plot_data['AUC'].tolist()
params = [i/100 for i in plot_data['parameters'].tolist()]

# Identify the best three models
top_models = methods[:3]  # Top 3 methods with highest AUC

# Colors for the top three models
top_colors = ['gold', 'orange', 'lightgreen']

# Colormap setup for the rest of the models
cmap = plt.colormaps['Blues']
colors = cmap(np.linspace(0.3, 1, len(methods))[::-1])  

# Create plot
fig, ax = plt.subplots(facecolor='white')

x_points = []
y_points = []
offset_x = 0.2
offset_y = 0.2
font_size = 8  

param_threshold = 10000

for i, (method, time, metric, color, param) in enumerate(list(zip(methods, inference_time, MRR_val, colors, params))):
    if i % 2 == 0: 
        ax.text(i - offset_x, metric - offset_y, f'{method}', fontsize=font_size, ha='left', va='bottom')
    
    # Set the color for the top models
    if method in top_models:
        color = top_colors[top_models.index(method)]
    
    if param < param_threshold:
        alpha_value = 0.4  # Less transparent
    else:
        alpha_value = 0.1 # More transparent
    
    ax.scatter(i, metric, s=param, color=color, alpha=alpha_value, edgecolor='b', linewidth=2)
    x_points.append(i)
    y_points.append(metric)

# Adding red "x" markers
for i, (method, time, metric, color, param) in enumerate(list(zip(methods, inference_time, MRR_val, colors, params))):
    ax.scatter(i, metric, s=10, color='red', marker='x', label=method)

# Add vertical lines at specific positions with spacing and zorder
vertical_lines = {
    'PageRank': 'DeepWalk', 
    'Node2vec': 'SAGE', 
    'GCN': 'SEAL', 
    'ELPH': 'Minilm', 
    'Llama': 'FT-bert'
}

# y_list = [[0.4, 0.6], [0.5, 0.8],  [0.7, 0.8], [0.7, 0.9], [0.9, 0.95]]
# for (start, end), y in zip(vertical_lines.items(), y_list):
#     start_idx = methods.index(start)
#     end_idx = methods.index(end)
#     mid_point = (start_idx + end_idx) / 2
#     ax.axvline(x=mid_point+0.3, color='gray', linestyle='--', linewidth=1.5,
#                ymin=y[0], ymax=y[1], zorder=10)  # Use zorder to bring the line to the front

# Customize x-ticks
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=12)

ax.set_ylabel('AUC (\%)', fontsize=14)
ax.grid(True, which="both", ls="--")

# Tick parameters
ax.tick_params(axis='both', which='both', direction='in', length=6)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=8)

# Title
ax.set_title('Comparison of Different Methods', fontsize=16)

# Adjust layout and save plot
plt.tight_layout()
plt.savefig('MRR1_plot_benchmark.pdf')