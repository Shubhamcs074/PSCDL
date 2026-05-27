import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({'font.family': 'sans-serif'})

# Model Metadata Specs
models = ['GeSCF', 'RSCD', 'C-3PO', 'Unified PSCDL Net (Ours)']
params_m = [12.4, 25.6, 48.2, 38.5] # Parameter footprint sizes
f1_scores = [0.4000, 0.5400, 0.6600, 0.7418]
colors = ['#A0A0A0', '#708090', '#4682B4', '#1E3A8A']
sizes = [150, 250, 400, 700] # Scale bubble sizes to match performance footprint importance

plt.figure(figsize=(8, 5.5))

for i, model in enumerate(models):
    plt.scatter(params_m[i], f1_scores[i], color=colors[i], s=sizes[i], 
                alpha=0.85, edgecolors='black', linewidths=1.5, label=model)
    
    # Offsetting annotations to ensure clean legibility
    plt.text(params_m[i] + 1.2, f1_scores[i] - 0.005, model, 
             fontweight='bold' if "Ours" in model else 'normal', fontsize=10)

plt.xlim(5, 55)
plt.ylim(0.3, 0.85)
plt.xlabel("Computational Footprint (Total Parameters in Millions)", fontweight='medium', labelpad=10)
plt.ylabel("Model Accuracy (Mean F1-Score Metric)", fontweight='medium', labelpad=10)
plt.title("Pareto Frontier: Parameter Efficiency vs. Downstream Performance", fontsize=13, pad=15, fontweight='bold')

plt.tight_layout()
plt.savefig(r"C:\Users\HP\Desktop\paper1\architectural_efficiency_scatter.png", dpi=300)
plt.close()
print("✨ Efficiency scatter diagram successfully deployed to disk asset folder.")