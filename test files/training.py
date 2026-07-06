# ==========================================
# MASTER PRODUCTION PIPELINE - MULTI-EPOCH RUNNER
# ==========================================
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from transformers import BertTokenizer, BertModel
from PIL import Image
import numpy as np

# 1. HARDWARE & PATH CONFIGURATION
PROJECT_LOCAL_PATH = r"C:\Users\HP\Desktop\PSCD"
CHECKPOINT_DIR = r"C:\Users\HP\Desktop\paper1\checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using Hardware Device: {device}")

# 2. LOCAL DATA ENGINE
class LocalPSCDLDataset(Dataset):
    def __init__(self, base_dir, img_size=(256, 256)):
        self.base_dir = base_dir
        self.ref_dir = os.path.join(self.base_dir, 't0')   
        self.curr_dir = os.path.join(self.base_dir, 't1')  
        self.mask_dir = os.path.join(self.base_dir, 'mask') 
        
        if os.path.exists(self.curr_dir):
            self.filenames = sorted([f for f in os.listdir(self.curr_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"✅ Data Link Established! Found {len(self.filenames)} local image pairs.")
        else:
            self.filenames = []
            print(f"❌ Local path verification failed! Cannot find: {self.curr_dir}")
            
        self.img_size = img_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.img_transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        json_path = os.path.join(base_dir, 'labels_PSCD_reduced_11to8.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                self.meta_labels = json.load(f)
        else:
            self.meta_labels = None

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        ref_img = self.img_transform(Image.open(os.path.join(self.ref_dir, filename)).convert('RGB'))
        curr_img = self.img_transform(Image.open(os.path.join(self.curr_dir, filename)).convert('RGB'))
        
        mask_raw = Image.open(os.path.join(self.mask_dir, filename)).convert('L')
        mask_img = transforms.Compose([transforms.Resize(self.img_size), transforms.ToTensor()])(mask_raw)
        mask_img = (mask_img > 0.5).float()
        
        text_prompt = "unattended objects, debris accumulation, graffiti, temporary encroachments"
        if self.meta_labels and filename in self.meta_labels:
            text_prompt = str(self.meta_labels[filename])
            
        text_inputs = self.tokenizer(text_prompt, padding='max_length', max_length=32, return_tensors="pt")
        return ref_img, curr_img, text_inputs['input_ids'].squeeze(0), text_inputs['attention_mask'].squeeze(0), mask_img, filename

# 3. COMPONENT BLOCKS
class TwinFeatureInteraction(nn.Module):
    def __init__(self, in_channels):
        super(TwinFeatureInteraction, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, ref, curr):
        diff = torch.abs(curr - ref)
        return self.conv(torch.cat([diff, curr], dim=1))

class SubspaceBackgroundFilter(nn.Module):
    def __init__(self, in_channels):
        super(SubspaceBackgroundFilter, self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.alpha = nn.Parameter(torch.zeros(1))
    def forward(self, ref, curr):
        subspace = self.proj(ref)
        return curr + self.alpha * (curr - subspace)

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=4):
        super(CrossModalAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, visual_tokens, text_tokens):
        out, _ = self.attn(query=visual_tokens, key=text_tokens, value=text_tokens)
        return self.norm(visual_tokens + out)

class ProgressiveRefinementBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProgressiveRefinementBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, deep_features, lateral_skip=None):
        x = self.up(deep_features)
        if lateral_skip is not None:
            x = x + lateral_skip
        return self.refine(x)

# 4. SOTA UNIFIED NETWORK ARCHITECTURE
class SOTA_Unified_PSCDL_Net(nn.Module):
    def __init__(self):
        super(SOTA_Unified_PSCDL_Net, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.enc_init = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  
        self.layer2 = resnet.layer2  
        
        self.twin_module = TwinFeatureInteraction(in_channels=512)
        self.subspace_filter = SubspaceBackgroundFilter(in_channels=512)
        
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert.parameters():
            param.requires_grad = False
        self.text_projector = nn.Linear(768, 512)
        self.cross_modal_alignment = CrossModalAttention(embed_dim=512, num_heads=4)
        
        self.stage1_fusion = nn.Conv2d(512 * 2, 512, kernel_size=1)
        self.stage2_refine = ProgressiveRefinementBlock(in_channels=512, out_channels=256)
        self.stage3_refine = ProgressiveRefinementBlock(in_channels=256, out_channels=64)
        
        self.final_segmentation_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, ref_img, curr_img, text_ids, text_mask):
        B, C, H, W = curr_img.shape
        r_init, c_init = self.enc_init(ref_img), self.enc_init(curr_img)
        r_l1, c_l1 = self.layer1(r_init), self.layer1(c_init)  
        r_l2, c_l2 = self.layer2(r_l1), self.layer2(c_l1)      
        
        twin_mapped_curr = self.twin_module(r_l2, c_l2)
        subspace_filtered_curr = self.subspace_filter(r_l2, twin_mapped_curr)
        
        with torch.no_grad():
            bert_out = self.bert(input_ids=text_ids, attention_mask=text_mask)
            text_context = bert_out.last_hidden_state
        text_tokens = self.text_projector(text_context)
        
        FH, FW = subspace_filtered_curr.shape[2], subspace_filtered_curr.shape[3]
        spatial_tokens = subspace_filtered_curr.flatten(2).transpose(1, 2)
        
        semantic_visual_tokens = self.cross_modal_alignment(spatial_tokens, text_tokens)
        curr_l2_semantic = semantic_visual_tokens.transpose(1, 2).view(B, 512, FH, FW)
        
        coarse_fused = self.stage1_fusion(torch.cat([r_l2, curr_l2_semantic], dim=1))
        refined_mid = self.stage2_refine(coarse_fused, lateral_skip=c_l1) 
        refined_fine = self.stage3_refine(refined_mid)
        
        output_mask = self.final_segmentation_head(refined_fine)
        return F.interpolate(output_mask, size=(H, W), mode='bilinear', align_corners=False)

# 5. LOSS FUNCTION
class SOTA_Pixel_F1_Loss(nn.Module):
    def __init__(self):
        super(SOTA_Pixel_F1_Loss, self).__init__()
        self.bce = nn.BCELoss()
    def forward(self, pred, target, smooth=1e-6):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        bce = self.bce(pred_flat, target_flat)
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 0.2 * bce + 0.8 * (1.0 - dice_score)

# 6. EVALUATION METRICS ENGINE
def evaluate_validation_set(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    all_f1s = []
    all_ious = []
    
    with torch.no_grad():
        for ref, curr, txt_ids, txt_msk, mask, _ in val_loader:
            ref, curr = ref.to(device), curr.to(device)
            txt_ids, txt_msk, mask = txt_ids.to(device), txt_msk.to(device), mask.to(device)
            
            preds = model(ref, curr, txt_ids, txt_msk)
            loss = criterion(preds, mask)
            val_loss += loss.item()
            
            # Convert continuous model outputs into hard binary pixels
            pred_bin = (preds > 0.5).float().view(-1)
            target_bin = mask.view(-1)
            
            intersection = (pred_bin * target_bin).sum().item()
            total_pixels = pred_bin.sum().item() + target_bin.sum().item()
            union = (pred_bin + target_bin > 0.5).float().sum().item()
            
            # Compute operational validation metrics safely
            f1 = (2.0 * intersection) / total_pixels if total_pixels > 0 else 1.0
            iou = intersection / union if union > 0 else 1.0
            
            all_f1s.append(f1)
            all_ious.append(iou)
            
    return val_loss / len(val_loader), np.mean(all_f1s), np.mean(all_ious)

# 7. EXECUTION ENGINE
if __name__ == '__main__':
    full_dataset = LocalPSCDLDataset(base_dir=PROJECT_LOCAL_PATH)
    
    if len(full_dataset) > 0:
        # Create an 85/15 validation split to protect generalization limits
        val_size = int(len(full_dataset) * 0.15)
        train_size = len(full_dataset) - val_size
        train_set, val_set = random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_set, batch_size=4, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=4, shuffle=False, drop_last=False)
        
        model = SOTA_Unified_PSCDL_Net().to(device)
        criterion = SOTA_Pixel_F1_Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        
        NUM_EPOCHS = 5
        best_val_loss = float('inf')
        
        print(f"🚀 Initializing Unified Training Pipeline: {train_size} training samples | {val_size} validation samples.")
        
        for epoch in range(NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            print(f"\n--- Starting Epoch {epoch+1}/{NUM_EPOCHS} ---")
            
            for batch_idx, (ref, curr, txt_ids, txt_msk, mask, _) in enumerate(train_loader):
                ref, curr = ref.to(device), curr.to(device)
                txt_ids, txt_msk, mask = txt_ids.to(device), txt_msk.to(device), mask.to(device)
                
                optimizer.zero_grad()
                predictions = model(ref, curr, txt_ids, txt_msk)
                loss = criterion(predictions, mask)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                if batch_idx % 20 == 0:
                    print(f"Batch: {batch_idx}/{len(train_loader)} | Running Dice Loss: {loss.item():.4f}")
            
            epoch_train_loss = running_loss / len(train_loader)
            epoch_val_loss, val_f1, val_iou = evaluate_validation_set(model, val_loader, criterion, device)
            
            print(f"✨ Epoch {epoch+1} Summary:")
            print(f"  -> Mean Train Loss: {epoch_train_loss:.4f}")
            print(f"  -> Mean Val Loss:   {epoch_val_loss:.4f}")
            print(f"  -> Val Mean F1-Score: {val_f1:.4f} | Val Mean IoU: {val_iou:.4f}")
            
            # Save the model whenever validation performance improves
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_pscdl_model.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_f1': val_f1
                }, checkpoint_path)
                print(f"💾 Saved superior checkpoint artifact to: {checkpoint_path}")
                
        print("\n🏆 Training optimization pipeline execution complete!")
    else:
        print("❌ Pipeline failed to start. Verify your dataset path directories.")