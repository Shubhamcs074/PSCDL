# ==========================================================
# PSCDL 2026 OFFICIAL VIDEO TEST SET SUBMISSION GENERATOR
# ==========================================================
import os
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import BertTokenizer

# Import your SOTA network architecture
from training import SOTA_Unified_PSCDL_Net

# PATH SETUP
TEST_VIDEOS_DIR = r"C:\Users\HP\Desktop\paper1\PSCDL2026_Test\test_videos"
SUBMISSION_BASE_DIR = r"C:\Users\HP\Desktop\paper1\submission_masks"
CHECKPOINT_PATH = r"C:\Users\HP\Desktop\paper1\checkpoints\best_pscdl_model.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_test_videos():
    print("⏳ Initializing Video Submission Generator Agent...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Reconstruct network and load your best weights
    model = SOTA_Unified_PSCDL_Net().to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"✅ Loaded optimized model from Epoch {checkpoint['epoch']}")
    
    # Core Image Transformations
    img_size = (256, 256)
    img_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Unified text prompt template 
    text_prompt = "unattended objects, debris accumulation, graffiti, temporary encroachments"
    text_inputs = tokenizer(text_prompt, padding='max_length', max_length=32, return_tensors="pt")
    txt_ids = text_inputs['input_ids'].to(device)
    txt_msk = text_inputs['attention_mask'].to(device)

    # Find video files
    video_files = sorted([f for f in os.listdir(TEST_VIDEOS_DIR) if f.lower().endswith('.mp4')])
    print(f"🚀 Found {len(video_files)} test videos to process.")

    with torch.no_grad():
        for video_name in video_files:
            video_path = os.path.join(TEST_VIDEOS_DIR, video_name)
            video_id = os.path.splitext(video_name)[0]
            
            # Create a unique output subdirectory for each video's frames
            video_output_dir = os.path.join(SUBMISSION_BASE_DIR, video_id)
            os.makedirs(video_output_dir, exist_ok=True)
            
            print(f"\n🎬 Opening stream: {video_name}")
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"❌ Failed to open video: {video_path}")
                continue
                
            # Step A: Capture the first frame as your Baseline Reference Frame (t0)
            ret, first_frame = cap.read()
            if not ret:
                print(f"❌ Empty video file: {video_name}")
                cap.release()
                continue
                
            # Convert BGR (OpenCV format) to RGB (PIL format)
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            ref_pil = Image.fromarray(first_frame_rgb)
            ref_tensor = img_transform(ref_pil).unsqueeze(0).to(device)
            
            orig_h, orig_w, _ = first_frame.shape
            frame_idx = 0
            
            # Step B: Loop through all subsequent frames (t1)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break # End of video
                
                frame_idx += 1
                
                # Process every frame (or skip frames if the video frame rate is very high)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                curr_pil = Image.fromarray(frame_rgb)
                curr_tensor = img_transform(curr_pil).unsqueeze(0).to(device)
                
                # Predict change detection mask
                pred = model(ref_tensor, curr_tensor, txt_ids, txt_msk)
                
                # Resize output back to original video dimensions
                pred_resized = torch.nn.functional.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
                pred_np = pred_resized.squeeze().cpu().numpy()
                
                # Threshold to strict binary pixels (0 or 255)
                binary_mask = (pred_np > 0.5).astype('uint8') * 255
                
                # Save the mask as a lossless PNG using a zero-padded naming convention
                mask_filename = f"frame_{frame_idx:05d}.png"
                output_path = os.path.join(video_output_dir, mask_filename)
                Image.fromarray(binary_mask).save(output_path)
                
                if frame_idx % 100 == 0:
                    print(f"  -> Processed {frame_idx} frames...")
                    
            cap.release()
            print(f"✅ Finished processing {video_id}. Masks saved to {video_output_dir}")
            
    print(f"\n🏆 All test videos successfully converted to submission formats inside: {SUBMISSION_BASE_DIR}")

if __name__ == '__main__':
    process_test_videos()