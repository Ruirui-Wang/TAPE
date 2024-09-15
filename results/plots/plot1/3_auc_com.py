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
    
#     # Convert the 'AUC' and 'MRR' columns to numerical values by removing the "±" symbol and converting to float
#     data['AUC'] = data['AUC'].str.split(' ± ').str[0].astype(float)
#     data['MRR'] = data['MRR'].str.split(' ± ').str[0].astype(float)
    
#     return data

# # Load and process the data
# file_path = '/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/results/plots/plot1/complexity_cora.csv'
# data = load_data(file_path)

# # Sort the DataFrame by the 'AUC' column in increasing order
# sorted_data = data.sort_values(by='AUC')

# # Reset index after sorting
# sorted_data.reset_index(drop=True, inplace=True)

# print(sorted_data)
# # Prepare the data for plotting
# plot_data = sorted_data[['method', 'parameters', 'inference time', 'AUC', 'MRR']]

# methods = plot_data['method'].tolist()
# inference_time = plot_data['inference time'].tolist()
# MRR_val = plot_data['MRR'].tolist()
# AUC_val = plot_data['AUC'].tolist()
# params = plot_data['parameters'].tolist()

# # Normalize inference_time to [0, 1]
# norm_time = (inference_time - np.min(inference_time)) / (np.max(inference_time) - np.min(inference_time))

# # Set up the colormap
# cmap = plt.colormaps['Blues']
# colors = cmap(np.linspace(0.3, 1, len(methods)))  

# fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')  # Use the same figsize
# x_points = []
# y_points = []

# # Adjust offsets and reduce font size for better readability
# offset_x = 0.5
# offset_y = 0.5
# font_size = 10

# for i, (method, time, metric, color, param) in enumerate(list(zip(methods, norm_time, AUC_val, colors, params))):
#     log_param = np.log(param) if param > 0 else 0  
    
#     if method == 'NCNC':
#         ax.text(metric - offset_x, log_param + offset_y * 2, f'{method}', fontsize=font_size, ha='left', va='bottom')
#         bubble_color = 'orange'  
#     elif method == 'NCN':
#         ax.text(metric - offset_x, log_param - offset_y * 2, f'{method}', fontsize=font_size, ha='left', va='bottom')
#         bubble_color = 'orange'  
#     elif i > 0 and abs(metric - x_points[-1]) < 0.04:
#         ax.text(metric + offset_x, log_param + offset_y, f'{method}', fontsize=font_size, ha='left', va='bottom')
#         bubble_color = color 
#     else:
#         ax.text(metric - offset_x, log_param - offset_y, f'{method}', fontsize=font_size, ha='left', va='bottom')
#         bubble_color = color 
    
#     bubble_size = time * 100000
#     alpha_value = 0.1 if bubble_size > 10000 else 0.5
#     edge_color = 'w' if bubble_size > 10000 else 'b'
    
#     ax.scatter(metric, log_param, s=bubble_size, color=bubble_color, alpha=alpha_value, linewidth=2)
#     x_points.append(metric)
#     y_points.append(log_param)

# # Add red "x" markers
# for i, (method, time, metric, color, param) in enumerate(list(zip(methods, norm_time, AUC_val, colors, params))):
#     log_param = np.log(param) if param > 0 else 0
#     ax.scatter(metric, log_param, s=10, color='red', marker='x', label=method)

# ax.set_xlabel('AUC (\%)', fontsize=14)
# ax.set_ylabel('Number of Params (Log Scale)', fontsize=14)
# ax.grid(True, which="both", ls="--")

# # Tick parameters
# ax.tick_params(axis='both', which='both', direction='in', length=6)
# ax.tick_params(axis='both', which='major', labelsize=12)
# ax.tick_params(axis='both', which='minor', labelsize=8)

# # Title
# ax.set_title('Comparison of Different Methods', fontsize=16)

# # Adjust layout and save plot
# plt.tight_layout()
# plt.savefig('AUC_plot_benchmark.pdf')

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
    
    # Convert the 'AUC' and 'MRR' columns to numerical values by removing the "±" symbol and converting to float
    data['AUC'] = data['AUC'].str.split(' ± ').str[0].astype(float)
    data['MRR'] = data['MRR'].str.split(' ± ').str[0].astype(float)
    
    return data

# Load and process the data
file_path = '/hkfs/work/workspace/scratch/cc7738-iclr25/cc7738-benchmark_tag/TAPE_chen/results/plots/plot1/complexity_cora.csv'
data = load_data(file_path)

# Sort the DataFrame by the 'AUC' column in increasing order
sorted_data = data.sort_values(by='AUC')

# Reset index after sorting
sorted_data.reset_index(drop=True, inplace=True)

print(sorted_data)
# Prepare the data for plotting
plot_data = sorted_data[['method', 'parameters', 'inference time', 'AUC', 'MRR']]

methods = plot_data['method'].tolist()
inference_time = plot_data['inference time'].tolist()
MRR_val = plot_data['MRR'].tolist()
AUC_val = plot_data['AUC'].tolist()
params = plot_data['parameters'].tolist()

# Normalize inference_time to [0, 1]
norm_time = (inference_time - np.min(inference_time)) / (np.max(inference_time) - np.min(inference_time))

# Set up the colormap
cmap = plt.colormaps['Blues']
colors = cmap(np.linspace(0.3, 1, len(methods)))  

# Adjust figure size to make the plot more compact
fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')  

x_points = []
y_points = []

# Reduce offsets for less whitespace
offset_x = 0.3
offset_y = 0.3
font_size = 10

for i, (method, time, metric, color, param) in enumerate(list(zip(methods, norm_time, AUC_val, colors, params))):
    log_param = np.log(param) if param > 0 else 0  
    
    if method == 'NCNC':
        ax.text(metric - offset_x, log_param + offset_y * 2, f'{method}', fontsize=font_size, ha='left', va='bottom')
        bubble_color = 'orange'  
    elif method == 'NCN':
        ax.text(metric - offset_x, log_param - offset_y * 2, f'{method}', fontsize=font_size, ha='left', va='bottom')
        bubble_color = 'orange'  
    elif i > 0 and abs(metric - x_points[-1]) < 0.04:
        ax.text(metric + offset_x, log_param + offset_y, f'{method}', fontsize=font_size, ha='left', va='bottom')
        bubble_color = color 
    else:
        ax.text(metric - offset_x, log_param - offset_y, f'{method}', fontsize=font_size, ha='left', va='bottom')
        bubble_color = color 
    
    bubble_size = time * 150000  # Increased bubble size for better visibility
    alpha_value = 0.1 if bubble_size > 10000 else 0.5
    edge_color = 'w' if bubble_size > 10000 else 'b'
    
    ax.scatter(metric, log_param, s=bubble_size, color=bubble_color, alpha=alpha_value, linewidth=2)
    x_points.append(metric)
    y_points.append(log_param)

# Add red "x" markers
for i, (method, time, metric, color, param) in enumerate(list(zip(methods, norm_time, AUC_val, colors, params))):
    log_param = np.log(param) if param > 0 else 0
    ax.scatter(metric, log_param, s=10, color='red', marker='x', label=method)

ax.set_xlabel('AUC (\%)', fontsize=14)
ax.set_ylabel('Number of Params (Log Scale)', fontsize=14)
ax.grid(True, which="both", ls="--")

# Tick parameters
ax.tick_params(axis='both', which='both', direction='in', length=6)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.tick_params(axis='both', which='minor', labelsize=8)

# Title
ax.set_title('Comparison of Different Methods', fontsize=16)

# Adjust layout and save plot
plt.tight_layout()
plt.savefig('AUC_plot_benchmark.pdf')