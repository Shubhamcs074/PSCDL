# ==========================================
# INFERENCE & VISUALIZATION AGENT
# ==========================================
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Import model architecture directly from your training code
from training import SOTA_Unified_PSCDL_Net, LocalPSCDLDataset

# PATH CONFIGURATIONS
PROJECT_LOCAL_PATH = r"C:\Users\HP\Desktop\PSCD"
CHECKPOINT_PATH = r"C:\Users\HP\Desktop\paper1\checkpoints\best_pscdl_model.pth"
OUTPUT_VIS_DIR = r"C:\Users\HP\Desktop\paper1\predictions"
os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_visual_inference(num_samples=3):
    print("⏳ Loading dataset configuration...")
    dataset = LocalPSCDLDataset(base_dir=PROJECT_LOCAL_PATH)
    
    print("💾 Loading optimized network weights...")
    model = SOTA_Unified_PSCDL_Net().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Model loaded successfully (Saved from Epoch {checkpoint['epoch']} with Val F1: {checkpoint['val_f1']:.4f})")
    
    # Pick sample indices uniformly across your dataset to visualize
    sample_indices = np.linspace(0, len(dataset) - 1, num_samples, dtype=int)
    
    with torch.no_grad():
        for idx in sample_indices:
            ref_img, curr_img, txt_ids, txt_msk, mask, filename = dataset[idx]
            
            # Prepare tensors for batch format expected by PyTorch
            ref_tensor = ref_img.unsqueeze(0).to(device)
            curr_tensor = curr_img.unsqueeze(0).to(device)
            txt_ids_tensor = txt_ids.unsqueeze(0).to(device)
            txt_msk_tensor = txt_msk.unsqueeze(0).to(device)
            
            # Forward pass to predict changes
            prediction_mask = model(ref_tensor, curr_tensor, txt_ids_tensor, txt_msk_tensor)
            prediction_mask = prediction_mask.squeeze(0).squeeze(0).cpu().numpy()
            
            # Convert continuous probability values into hard binary predictions
            binary_prediction = (prediction_mask > 0.5).astype(np.uint8) * 255
            
            # Un-normalize visual tensor channels back to standard RGB image arrays
            def denormalize(tensor):
                img = tensor.permute(1, 2, 0).cpu().numpy()
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                return np.clip(img, 0, 1)
            
            ref_vis = denormalize(ref_img)
            curr_vis = denormalize(curr_img)
            gt_mask_vis = mask.squeeze(0).cpu().numpy()
            
            # Plot the qualitative comparison layout
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(ref_vis)
            axes[0].set_title("Reference (t0)")
            axes[0].axis('off')
            
            axes[1].imshow(curr_vis)
            axes[1].set_title("Current (t1)")
            axes[1].axis('off')
            
            axes[2].imshow(gt_mask_vis, cmap='gray')
            axes[2].set_title("Ground Truth Mask")
            axes[2].axis('off')
            
            axes[3].imshow(binary_prediction, cmap='gray')
            axes[3].set_title("Model Prediction")
            axes[3].axis('off')
            
            output_plot_path = os.path.join(OUTPUT_VIS_DIR, f"result_{filename}")
            plt.tight_layout()
            plt.savefig(output_plot_path, bbox_inches='tight', dpi=150)
            plt.close()
            print(f"🖼️ Saved qualitative comparison map for image pairs to: {output_plot_path}")

if __name__ == '__main__':
    run_visual_inference(num_samples=5)