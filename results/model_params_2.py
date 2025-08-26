import matplotlib.pyplot as plt
import seaborn as sns
fontsize  = 16
markersize = 14
# Data points for two model families
e5_params = [0.5, 1, 3]  # Number of parameters (in billions)
e5_accuracy = [94.79, 96.78, 98.78]  # Corresponding accuracy values
minilm_std =  [0.44, 0.96, 1.02]  # Corresponding accuracy values
minilm_params = [22, 33, 110]  # LLaMA/Mistral params (in billions)
minilm_accuracy = [97.79, 98.04, 98.89]  # Corresponding accuracy values
bert_params = [14, 66, 110]
bert_accuracy = [92.72, 94.79, 95.78]  # Corresponding accuracy values

# Model names
e5_labels = ["E5-Small", "E5-Base", "E5-Large"]
minilm_labels = ["MiniLM-L6-v2", "L12-v2", "MPNet"]
bert_labels = ["TniyBERT", "DistilBERT", "BERT-Base"]

# Create figure
plt.figure(figsize=(6, 5))
sns.set_style("whitegrid")

# Plot E5 models (green, squares, dashed)
plt.plot(e5_params, e5_accuracy, color='mediumseagreen', linestyle='--', marker='s', markersize=markersize, label="E5")

# Plot MiniLM/MPNet models (blue, circles, dashed)
plt.plot(minilm_params, minilm_accuracy, color='royalblue', linestyle='--', marker='o', markersize=markersize, label="MPNet/MiniLM")

# Plot Bert models (blue, stars, dashed)
plt.plot(bert_params, bert_accuracy, color='royalblue', linestyle='--', marker='*', markersize=markersize, label="BERT")

# Annotate points
# Annotate points with adjusted font size and position
for i, txt in enumerate(e5_labels):
    plt.annotate(txt, (e5_params[i], e5_accuracy[i]), 
                 textcoords="offset points", xytext=(-5, 5), ha='left', fontsize=fontsize)

for i, txt in enumerate(minilm_labels):
    if txt == 'L12-v2':
        plt.annotate(txt, (minilm_params[i], minilm_accuracy[i]), 
                textcoords="offset points", xytext=(25, 20), ha='left', fontsize=fontsize)
    else:
        plt.annotate(txt, (minilm_params[i], minilm_accuracy[i]), 
                    textcoords="offset points", xytext=(5, 5), ha='left', fontsize=fontsize)

for i, txt in enumerate(bert_labels):
    plt.annotate(txt, (bert_params[i], bert_accuracy[i]), 
                 textcoords="offset points", xytext=(5, 5), ha='left', fontsize=fontsize)

import matplotlib.pyplot as plt

plt.xticks(fontsize=fontsize)  # Change x-axis tick label size
plt.yticks(fontsize=fontsize)  # Change y-axis tick label size

# Alternative:
plt.tick_params(axis='both', labelsize=fontsize)  # Change both x and y axis tick label sizes

fontsize = 24
# Labels and title
plt.xlabel("Number of Parameters (Billion)", fontsize=fontsize)
plt.ylabel("AUC", fontsize=fontsize)
plt.ylim(90, 101)
plt.title("Effect of PLM Size on AUC", fontsize=fontsize)
plt.legend(loc="lower right", fontsize=12)

# Formatting
plt.grid(True)
plt.tight_layout()

# Save figure
root = '/hkfs/work/workspace/scratch/cc7738-rebuttal/TAPE_chen/TAPE/results'
plt.savefig(f"{root}/_model_params.png", dpi=300)
