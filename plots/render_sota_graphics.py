# ==========================================================
# ADVANCED HIGH-FIDELITY BENCHMARK VISUALIZATION DISPATCHER
# ==========================================================
import os
import numpy as np
import matplotlib.pyplot as plt

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('default')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.titlesize': 16
})

OUTPUT_PATH = r"C:\Users\HP\Desktop\paper1\benchmarks_sota.png"

def generate_production_graphics():
    print("🎨 Initializing Advanced Graphic Visualization Pipeline...")
    
    models = ['GeSCF', 'RSCD', 'C-3PO', 'Unified PSCDL Net (Ours)']
    metrics = ['Precision', 'Recall', 'F1-Score', 'IoU']
    
    data = {
        'GeSCF': [0.4310, 0.3732, 0.4000, 0.2800],
        'RSCD':  [0.5821, 0.5034, 0.5400, 0.4000],
        'C-3PO': [0.6944, 0.6291, 0.6600, 0.5400],
        'Unified PSCDL Net (Ours)': [0.7853, 0.7029, 0.7418, 0.5948]
    }
    
    colors = ['#A0A0A0', '#708090', '#4682B4', '#1E3A8A']
    fig = plt.figure(figsize=(15, 6.5))
    
    # PANEL A: MULTI-DIMENSIONAL RADAR CHART
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    ax1 = fig.add_subplot(121, polar=True)
    ax1.set_theta_offset(np.pi / 2)
    ax1.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], metrics, color='#333333', fontweight='semibold')
    ax1.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8], ["0.2", "0.4", "0.6", "0.8"], color="#999999", size=10)
    plt.ylim(0, 0.9)
    
    for model_name, color in zip(models, colors):
        values = data[model_name]
        values_closed = values + values[:1]
        
        linewidth = 3.2 if "Ours" in model_name else 1.5
        alpha = 0.22 if "Ours" in model_name else 0.04
        zorder = 5 if "Ours" in model_name else 2
        
        ax1.plot(angles, values_closed, color=color, linewidth=linewidth, label=model_name, zorder=zorder)
        ax1.fill(angles, values_closed, color=color, alpha=alpha, zorder=zorder - 1)
        
    ax1.set_title("A: Multi-Metric Capability Mapping", pad=20, fontweight='bold', color='#111111')
    ax1.legend(loc='upper right', bbox_to_anchor=(-0.1, 1.0))
    
    # PANEL B: COMPARATIVE HORIZONTAL BAR CHART
    ax2 = fig.add_subplot(122)
    f1_scores = [data[m][2] for m in models]
    
    bars = ax2.barh(models, f1_scores, color=colors, height=0.55, edgecolor='#333333', linewidth=0.8)
    ax2.set_xlim(0, 0.9)
    ax2.set_xlabel("Mean F1-Score Value (📌 Primary Challenge Metric)", labelpad=10, fontweight='medium')
    ax2.set_title("B: Leaderboard F1-Score Discrepancy", pad=20, fontweight='bold', color='#111111')
    
    for bar in bars:
        width = bar.get_width()
        is_ours = (bar.get_y() == bars[-1].get_y())
        fontweight = 'bold' if is_ours else 'normal'
        text_color = '#FFFFFF' if is_ours else '#222222'
        x_pos = width - 0.12 if is_ours else width + 0.01
        
        label_text = f"{width:.4f}"
        if is_ours:
            label_text += " (SOTA)"
            
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2, label_text, 
                 va='center', ha='left', color=text_color, fontweight=fontweight, size=11)
        
    c3po_f1 = data['C-3PO'][2]
    ours_f1 = data['Unified PSCDL Net (Ours)'][2]
    delta = ours_f1 - c3po_f1
    
    ax2.annotate('', xy=(c3po_f1, 2), xytext=(ours_f1, 3), 
                arrowprops=dict(arrowstyle="<->", color='#B22222', lw=1.5, ls='dashed'))
    ax2.text((c3po_f1 + ours_f1)/2 + 0.01, 2.5, f"+{delta*100:.1f}% Net Gain", 
             color='#B22222', fontweight='bold', ha='center', size=11)

    for spine in ['top', 'right']:
        ax2.spines[spine].set_visible(False)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✨ Production graphics rendered successfully and saved to: {OUTPUT_PATH}")

if __name__ == '__main__':
    generate_production_graphics()