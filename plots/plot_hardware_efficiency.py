import os
import matplotlib.pyplot as plt

try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('default')

plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 11})
OUTPUT_PATH = r"C:\Users\HP\Desktop\paper1\hardware_fps_throughput.png"

def generate_throughput_curves():
    print("📈 Compiling Computational Throughput Scalability Matrix Curves...")
    
    resolutions = ['256x256', '512x512', '1024x1024']
    
    # Inference speed throughput curves (FPS tracking frames per second metrics)
    fps_c3po = [45, 18, 4]  # Heavy architectural bottlenecks
    fps_ours = [118, 56, 19] # Our highly efficient pruned baseline pathway pipelines
    
    fig, ax = plt.subplots(figsize=(7.5, 5))
    
    ax.plot(resolutions, fps_ours, color='#1E3A8A', marker='o', linewidth=3, markersize=8, label='Unified PSCDL Net (Ours)')
    ax.plot(resolutions, fps_c3po, color='#4682B4', marker='s', linewidth=2, markersize=7, linestyle='--', label='C-3PO Baseline')
    
    # Structural layout labels
    ax.set_ylabel('Inference Execution Speed (Frames Per Second - FPS)', fontweight='medium', labelpad=10)
    ax.set_xlabel('Input Image Resolution Spatial Scaling Profile', fontweight='medium', labelpad=10)
    ax.set_title("Computational Footprint Scalability & Frame Throughput Efficiency", fontsize=13, pad=15, fontweight='bold')
    
    # Add real-time target guideline line at 30 FPS standard
    ax.axhline(y=30, color='#B22222', linestyle=':', alpha=0.7, linewidth=1.5)
    ax.text(1.8, 32, 'Real-time Standard Line (30 FPS)', color='#B22222', style='italic', weight='semibold', ha='right', size=9.5)
    
    ax.legend(loc='upper right', frameon=True)
    
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
        
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300)
    plt.close()
    print(f"✨ Throughput velocity comparison asset dispatched to: {OUTPUT_PATH}")

if __name__ == '__main__':
    generate_throughput_curves()