import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
# # Enable LaTeX formatting in Matplotlib
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'  # Use serif fonts
# plt.rcParams['text.latex.preamble'] = r'\usepackage{helvet}'  # Use Helvetica font
# Set up the theme and font
sns.set_theme(style="whitegrid", palette="pastel")
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Amiri"
})
# Function to extract the mean value from the string "mean ± std"
def extract_mean(value):
    try:
        mean_value = float(value.split(' ± ')[0])
    except:
        mean_value = None
    return mean_value

# Calculate group start indices dynamically
def calculate_group_start_indices(group_labels, bar_width, group_spacing_factor=7):
    current_position = 0
    group_start_indices = {}
    
    for group, models in group_labels.items():
        group_start_indices[group] = np.arange(len(models)) * (bar_width * 3) + current_position
        current_position = group_start_indices[group][-1] + bar_width * group_spacing_factor
    
    return group_start_indices

# Define the groups (including all available models)
group_labels = {
    'NCNC': [r"\textbf{bert-ncnc}", r"\textbf{minilm-ncnc}", r"\textbf{e5-ncnc}", r"\textbf{llama-ncnc}"],
    'Buddy': [r"\textbf{bert-buddy}", r"\textbf{minilm-buddy}", r"\textbf{e5-buddy}", r"\textbf{llama-buddy}", ],
    'NCN': [ r"\textbf{bert-ncn}", r"\textbf{minilm-ncn}", r"\textbf{e5-ncn}", r"\textbf{llama-ncn}"],
    'HLGNN': [ r"\textbf{bert-hlgnn}", r"\textbf{minilm-hlgnn}",  r"\textbf{e5-hlgnn}", r"\textbf{llama-hlgnn}",],
    'NeoGNN': [  r"\textbf{bert-neognn}", r"\textbf{minilm-neognn}", r"\textbf{e5-neognn}",  r"\textbf{llama-neognn}",]
}


for data_name in ['cora', 'arxiv_2023', 'pubmed']:
    
    file_path = f'/hkfs/work/workspace/scratch/cc7738-benchmark_tag/TAPE_chen/results/plots/plot2/{data_name}.csv'
    data_name = f'-{data_name}'
    data = pd.read_csv(file_path)
    
    # Apply the function to relevant columns
    for col in data.columns[1:]:  # Skip the 'Metric' column
        data[col] = data[col].apply(extract_mean)

    bar_width = 3.0  
    group_start_indices = calculate_group_start_indices(group_labels, bar_width)

    mrr_data = []
    hits50_data = []
    auc_data = []
    labels = []
    x_positions = []

    # Populate data for plotting
    for group, models in group_labels.items():
        for i, model in enumerate(models):
            mrr_value = data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + data_name, 'MRR'].values[0] if not data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + data_name, 'MRR'].empty else 0
            hits50_value = data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + data_name, 'Hits@50'].values[0] if not data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + data_name, 'Hits@50'].empty else 0
            auc_value = data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + data_name, 'AUC'].values[0] if not data.loc[data['Metric'] == model.split('{')[1].split('}')[0] + data_name, 'AUC'].empty else 0
            
            mrr_data.append(mrr_value)
            hits50_data.append(hits50_value)
            auc_data.append(auc_value)

            x_positions.append(group_start_indices[group][i])

    labels = [r"\textbf{Bert}",  r"\textbf{Minilm}",  r"\textbf{e5}", r"\textbf{Llama3}", ] * 5
    cmap = cm.get_cmap('Blues')
    norm = mcolors.Normalize(vmin=min(mrr_data + hits50_data + auc_data), vmax=max(mrr_data + hits50_data + auc_data))

