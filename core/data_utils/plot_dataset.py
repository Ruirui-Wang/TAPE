import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset from CSV
data = pd.read_csv('/hkfs/work/workspace/scratch/cc7738-iclr25/cc7738-benchmark_tag/TAPE_chen/core/data_utils/plot_data.csv')
import matplotlib.pyplot as plt
import pandas as pd


# Sample dataset (You can load this from a file or create it manually)
data = pd.DataFrame({
    'name': ['pwc_small', 'cora', 'pubmed', 'arxiv_2023', 'pwc_medium', 'ogbn-arxiv', 'citationv8', 'ogbn-paper100M',
             'ogbl-ppa', 'ogbl-collab', 'ogbl-ddi', 'ogbl-citation2', 'ogbl-wikikg2', 'ogbl-biokg', 'ogbl-vessel'],
    'num_nodes': [140, 2485, 19717, 13153, 86795, 169343, 1084224, 111059956, 576289, 235868, 4267, 2927963, 2500604, 93773, 3538495],
    'num_edges': [796, 10138, 88648, 68260, 933138, 2315598, 12198168, 1615685872, 30326273, 1285465, 1334889, 30561187, 17137181, 5088434, 5345897],
    'proximity': [0.4092, 0.1575, 0.2729, 0.8792, 0.4580, 0.8265, 0.4711, 0.5, 0.5, 0.9335, 0.5, 0.8631, 0.5, 0.5, 0.3112]
})

# Condition for "our datasets" (those that do not start with "ogbl")
ours = ~data['name'].str.startswith('ogbl')

ogbl_datasets = data['name'].str.startswith('ogbl') & ~data['name'].isin(['ogbl-ddi', 'ogbl-ppa'])

# Special cases for ddi and ppa (to be colored grey)
ddi_ppa = data['name'].isin(['ogbl-ddi', 'ogbl-ppa'])

# Size scaling for bubbles
ours_size = [int(i * 400) for i in data['proximity']]  # Scaling factor for bubble sizes

# Plotting
fig, ax = plt.subplots()

from adjustText import adjust_text

# Your scatter plot code goes here

ax.scatter(data['num_nodes'][ours], data['num_edges'][ours], 
           s=[ours_size[i] for i in range(len(ours_size)) if ours.iloc[i]], 
           c='orange', label="Our datasets", alpha=0.6)

# Scatter plot for "existing" datasets (those starting with 'ogbl')
ax.scatter(data['num_nodes'][ogbl_datasets], data['num_edges'][ogbl_datasets], 
           s=[ours_size[i] for i in range(len(ours_size)) if ogbl_datasets.iloc[i]], 
           c='#87CEEB', label="ogbl datasets", alpha=0.8)

ax.scatter(data['num_nodes'][ddi_ppa], data['num_edges'][ddi_ppa], c='grey', s=200, alpha=0.8)

import numpy as np
import matplotlib.pyplot as plt

# Your scatter plot code goes here (assumed)

# Apply random shifts to annotations manually


# # Log scales
ax.set_xscale('log')
ax.set_yscale('log')

# Labels and title
ax.set_xlabel('Number of nodes')
ax.set_ylabel('Number of edges')

# Legend
ax.legend()

# Save the plot to a file
# delete previous file first 
import os
if os.path.exists('draft_new_dataset_plot.pdf'):
    os.remove('draft_new_dataset_plot.pdf')
plt.savefig('draft_new_dataset_plot.pdf')

# Show plot
plt.show()

