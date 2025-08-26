import matplotlib.pyplot as plt
import seaborn as sns

# Data points for two model families
e5_params = [0.5, 1, 3]  # Number of parameters (in billions)
e5_accuracy = [94.79, 96.78, 98.78]  # Corresponding accuracy values
minilm_std =  [0.44, 0.96, 1.02]  # Corresponding accuracy values
minilm_params = [22, 33, 110]  # LLaMA/Mistral params (in billions)
minilm_accuracy = [97.79, 98.04, 98.89]  # Corresponding accuracy values
# [97.79 ± 0.44, 98.04 ± 0.96, 98.89 ± 1.23] 
bert_params = [14, 66, 110]
bert_accuracy = [92.72, 94.79, 95.78]  # Corresponding accuracy values


# Model names
e5_labels = ["E5-Small", "E5-Base", "E5-Large"]
minilm_labels = ["MiniLM-L6-v2", "MiniLM-L12-v2", "MPNet"]
bert_labels = ["TniyBERT", "DistilBERT", "BERT-Base"]

# Create figure
plt.figure(figsize=(6, 5))
import numpy as np
# Plot E5 models (green, squares, dashed)
plt.plot(e5_params, e5_accuracy, color='mediumseagreen', linestyle='--', marker='s', markersize=8, label="E5 Models")

plt.plot(minilm_params, minilm_accuracy, color='royalblue', linestyle='--', marker='o', markersize=6, label="MPNet/MiniLM Models")
# plt.fill_between(minilm_params, np.array(minilm_accuracy) - np.array(minilm_std), np.array(minilm_accuracy) + np.array(minilm_std), 
#                  color='royalblue', alpha=0.2)
# Plot LLaMA models (blue, circles, dashed)

plt.plot(bert_params, bert_accuracy, color='royalblue', linestyle='--', marker='*', markersize=6, label="Bert Models")



# Annotate points
for i, txt in enumerate(e5_labels):
    plt.annotate(txt, (e5_params[i], e5_accuracy[i]), textcoords="offset points", xytext=(-5, 5), ha='left')

for i, txt in enumerate(minilm_labels):
    plt.annotate(txt, (minilm_params[i], minilm_accuracy[i]), textcoords="offset points", xytext=(-5, 5), ha='left')

for i, txt in enumerate(bert_labels):
    plt.annotate(txt, (bert_params[i], bert_accuracy[i]), textcoords="offset points", xytext=(5, 5), ha='left')


# Labels and title
plt.xlabel("Number of Parameters (Billion)")
plt.ylabel("AUC")
plt.ylim(90, 101)
plt.title("Effect of Model Size on AUC")
plt.legend()
plt.grid(True)


# Save figure
root = '/hkfs/work/workspace/scratch/cc7738-rebuttal/TAPE_chen/TAPE/results'
plt.savefig(f"{root}_model_params.png")
