# Plot: Degradation of models on specific perspective 
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
models = ['S-LLAMA-1.3B', 'LLAMA-2-7B', 'Mistral-7B']
uni = [90, 89, 90]
bi = [85, 82, 85]
bi_mntp = [91, 90, 91]
bi_mntp_simcse = [90, 90, 90]

# Bar width and positions
bar_width = 0.2
index = np.arange(len(models))

# Create figure and axes
fig, ax = plt.subplots()

# Plot bars
bars_uni = ax.bar(index - 1.5*bar_width, uni, bar_width, label='Uni', color='white', edgecolor='black')
bars_bi = ax.bar(index - 0.5*bar_width, bi, bar_width, label='Bi', color='teal')
bars_bi_mntp = ax.bar(index + 0.5*bar_width, bi_mntp, bar_width, label='Bi + MNTP', color='gold')
bars_bi_mntp_simcse = ax.bar(index + 1.5*bar_width, bi_mntp_simcse, bar_width, label='Bi + MNTP + SimCSE', color='purple')

# Horizontal line at 85
ax.axhline(y=85, color='black', linestyle='--')

# Customizing the plot
ax.set_xlabel('')
ax.set_ylabel('Accuracy')
ax.set_title('(a) Chunking')
ax.set_xticks(index)
ax.set_xticklabels(models)
ax.set_ylim(75, 100)
ax.legend()

# Display plot
plt.savefig('results/plots/degradation/plot3.png')