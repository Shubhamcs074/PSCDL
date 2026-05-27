# ==========================================================
# PSCDL 2026 OFFICIAL TEST SET SUBMISSION GENERATOR
# ==========================================================
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer

# Import your SOTA network architecture
from training import SOTA_Unified_PSCDL_Net

# SUBMISSION PATH CONFIGURATIONS
TEST_DATA_DIR = r"C:\Users\HP\Desktop\PSCD_Test"   # Update when test set is released
SUBMISSION_DIR = r"C:\Users\HP\Desktop\paper1\submission_masks"
CHECKPOINT_PATH = r"C:\Users\HP\Desktop\paper1\checkpoints\best_pscdl_model.pth"

os.makedirs(SUBMISSION_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def generate_submission():
    print("⏳ Initializing Submission Generator Agent...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Reconstruct network and load your best weights
    model = SOTA_Unified_PSCDL_Net().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Loaded optimized model from Epoch {checkpoint['epoch']}")
    
    # Setup standard test transforms (matching training image dimensions)
    img_size = (256, 256)
    img_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Read test frames (Assuming 't0' and 't1' folders match the training layout)
    t1_dir = os.path.join(TEST_DATA_DIR, 't1')
    t0_dir = os.path.join(TEST_DATA_DIR, 't0')
    
    test_files = sorted([f for f in os.listdir(t1_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"🚀 Processing {len(test_files)} testing pairs...")
    
    # Default fallback prompt if textual metadata changes or is omitted in test set
    text_prompt = "unattended objects, debris accumulation, graffiti, temporary encroachments"
    text_inputs = tokenizer(text_prompt, padding='max_length', max_length=32, return_tensors="pt")
    txt_ids = text_inputs['input_ids'].to(device)
    txt_msk = text_inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        for filename in test_files:
            # Load images
            ref_raw = Image.open(os.path.join(t0_dir, filename)).convert('RGB')
            curr_raw = Image.open(os.path.join(t1_dir, filename)).convert('RGB')
            orig_w, orig_h = curr_raw.size
            
            # Transform to model size
            ref_tensor = img_transform(ref_raw).unsqueeze(0).to(device)
            curr_tensor = img_transform(curr_raw).unsqueeze(0).to(device)
            
            # Predict change mask
            pred = model(ref_tensor, curr_tensor, txt_ids, txt_msk)
            
            # Resize the continuous mask back to the original image dimensions
            pred_resized = torch.nn.functional.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
            pred_np = pred_resized.squeeze().cpu().numpy()
            
            # Threshold to strict binary pixels (0 or 255) as per competition standards
            binary_mask = (pred_np > 0.5).astype('uint8') * 255
            
            # Save the raw mask image
            output_path = os.path.join(SUBMISSION_DIR, filename)
            Image.fromarray(binary_mask).save(output_path)
            
    print(f"🏆 Verification complete. All submission masks safely written to: {SUBMISSION_DIR}")

if __name__ == '__main__':
    generate_submission()