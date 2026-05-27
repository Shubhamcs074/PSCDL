import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})

epochs = [1, 2, 3, 4, 5]
train_loss = [0.684, 0.412, 0.295, 0.188, 0.104]
val_f1 = [0.5210, 0.6145, 0.6890, 0.7230, 0.7418]

fig, ax1 = plt.subplots(figsize=(8, 4.5))

# Plotting Loss Trajectory
color = '#B22222'
ax1.set_xlabel('Training Epochs', fontweight='semibold')
ax1.set_ylabel('Total Optimization Loss', color=color, fontweight='semibold')
line1 = ax1.plot(epochs, train_loss, color=color, marker='o', linewidth=2.5, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(epochs)

# Instantiating a secondary axes that shares the same x-axis
ax2 = ax1.twinx()  
color = '#1E3A8A'
ax2.set_ylabel('Validation F1-Score', color=color, fontweight='semibold')
line2 = ax2.plot(epochs, val_f1, color=color, marker='s', linewidth=2.5, linestyle='--', label='Val F1-Score')
ax2.tick_params(axis='y', labelcolor=color)

# Highlight SOTA Checkpoint Convergence Point
ax2.annotate('Best Model Checkpoint\nF1: 0.7418', xy=(5, 0.7418), xytext=(3.5, 0.65),
             arrowprops=dict(facecolor='#111111', shrink=0.08, width=1, headwidth=6),
             fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.3))

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='center left')

plt.title("Model Convergence Profile & Metric Optimization History", fontsize=13, pad=15, fontweight='bold')
plt.tight_layout()
plt.savefig(r"C:\Users\HP\Desktop\paper1\training_convergence.png", dpi=300)
plt.close()
print("✨ Convergence plot saved to desktop.")