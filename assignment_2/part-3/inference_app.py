import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms as T
import yaml
import os
from types import SimpleNamespace

# ---------------------------------------------------------
# 1. MODEL ARCHITECTURES
# ---------------------------------------------------------
def get_semantic_model(device):
    # Initialize an untrained model and update the number of classes to 22 (20 classes + 1 Background + 1 Border)
    model = deeplabv3_resnet50(weights=None)
    model.classifier[4] = torch.nn.Conv2d(256, 22, kernel_size=(1, 1), stride=(1, 1))
    model.to(device)
    return model

def get_instance_model(device):
    # Initialize an untrained model and update the number of classes to 2 (1 Foreground + 1 Background)
    model = maskrcnn_resnet50_fpn(weights=None)
    
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, 2)
    
    model.to(device)
    return model


# ---------------------------------------------------------
# 2. POST-PROCESSING LOGIC
# ---------------------------------------------------------
def decode_segmap(image, num_classes=21):
    # Standard 21-color palette of the Pascal VOC dataset
    label_colors = np.array([
        (0,0,0), (128,0,0), (0,128,0), (128,128,0), (0,0,128),
        (128,0,128), (0,128,128), (128,128,128), (64,0,0), (192,0,0),
        (64,128,0), (192,128,0), (64,0,128), (192,0,128), (64,128,128),
        (192,128,128), (0,64,0), (128,64,0), (0,192,0), (128,192,0),
        (0,64,128)
    ])
    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, num_classes):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    return np.stack([r, g, b], axis=2)


# ---------------------------------------------------------
# 3. APPLICATION WORKFLOW
# ---------------------------------------------------------
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Initializing model on device: {device}")
    
    # Open image using RGB color format
    try:
        img = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"[!] Unable to open image file '{args.image}'. Error: {e}")
        return
        
    print(f"[*] Image loaded successfully.")

    # ================= SEMANTIC SEGMENTATION PIPELINE =================
    if args.task == 'semantic':
        print("[*] Constructing Semantic Segmentation architecture...")
        checkpoint_path = os.path.join("models", "semantic_best_deeplabv3_voc.pth")
        model = get_semantic_model(device)
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False), strict=False)
        except FileNotFoundError:
            print(f"[!] Warning: Model checkpoint not found at '{checkpoint_path}'. Please ensure the model file is downloaded and placed in the 'models' directory.")
            return
        model.eval()
        
        print("[*] Step 1: Pre-processing raw image...")
        transform = T.Compose([
            T.Resize((256, 256)), # Alternatively, maintain original size and interpolate later
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        print("[*] Step 2: Executing model inference...")
        with torch.no_grad():
            output = model(img_tensor)['out']
            pred = output.argmax(dim=1)[0].cpu().numpy()
            
        print("[*] Step 3: Post-processing prediction into color mask...")
        color_mask = decode_segmap(pred)
        
        # Render visualization
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img.resize((256, 256)))
        axs[0].set_title("Input Image")
        axs[0].axis('off')
        
        axs[1].imshow(color_mask)
        axs[1].set_title("Segmentation Result")
        axs[1].axis('off')
        
        print("[*] Displaying and saving visualization results...")
        plt.tight_layout()
        out_dir = "outputs/semantic"
        os.makedirs(out_dir, exist_ok=True)
        out_name = os.path.join(out_dir, os.path.basename(args.image))
        plt.savefig(out_name, bbox_inches='tight')
        print(f"[+] Success! Output saved at: {out_name}")
        plt.show()
        
    # ================= INSTANCE SEGMENTATION PIPELINE =================
    elif args.task == 'instance':
        print("[*] Constructing Instance Segmentation architecture...")
        checkpoint_path = os.path.join("models", "instance_best_maskrcnn_voc.pth")
        model = get_instance_model(device)
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False), strict=False)
        except FileNotFoundError:
            print(f"[!] Warning: Model checkpoint not found at '{checkpoint_path}'. Please ensure the model file is downloaded and placed in the 'models' directory.")
            return
        model.eval()
        
        print("[*] Step 1: Pre-processing image...")
        img_tensor = T.functional.to_tensor(img).to(device)
        
        print("[*] Step 2: Executing model inference...")
        with torch.no_grad():
            pred = model([img_tensor])[0]
            
        print("[*] Step 3: Post-processing - Extracting bounding boxes and masks...")
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        masks = pred['masks'].cpu().numpy()
        
        # Initialize visualization plot
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(img)
        
        # Filter predictions (Retain boxes with confidence score > threshold)
        valid_idxs = np.where(scores > args.threshold)[0]
        
        import colorsys
        colors = [colorsys.hsv_to_rgb(i / max(1, len(valid_idxs)), 1, 1) for i in range(max(1, len(valid_idxs)))]
        
        for idx, obj_idx in enumerate(valid_idxs):
            box = boxes[obj_idx]
            mask = masks[obj_idx, 0] > 0.5
            c = colors[idx % len(colors)]
            
            # Draw bounding box rectangle
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], 
                                     linewidth=2, edgecolor=c, facecolor='none')
            ax.add_patch(rect)
            
            # Apply color to object mask
            colored_mask = np.zeros((*mask.shape, 3))
            for i in range(3): colored_mask[:, :, i] = c[i]
            
            # Overlay RGB color channels and utilize binary mask as alpha channel (50% transparency)
            img_mask_alpha = np.dstack([colored_mask, mask * 0.5]) 
            ax.imshow(img_mask_alpha)
            
            # Render confidence score label
            ax.text(box[0], box[1]-5, f'Confidence: {scores[obj_idx]:.2f}', color='white', 
                    bbox=dict(facecolor=c, alpha=0.5))
            
        ax.axis('off')
        plt.title("Instance Bounding Box & Alpha Mask Output")
        plt.tight_layout()
        out_dir = "outputs/instance"
        os.makedirs(out_dir, exist_ok=True)
        out_name = os.path.join(out_dir, os.path.basename(args.image))
        plt.savefig(out_name, bbox_inches='tight')
        print(f"[+] Success! Output saved at: {out_name}")
        print("[*] Displaying visualization results...")
        plt.show()

# Entry point for configuration-based execution
if __name__ == '__main__':
    print("[*] Reading configuration from 'config.yaml'...")
    try:
        with open('config.yaml', 'r', encoding='utf-8') as f:
            # Convert YAML directly into an args object to maintain backward compatibility
            config_dict = yaml.safe_load(f)
            args = SimpleNamespace(**config_dict) if config_dict else SimpleNamespace()
        main(args)
    except FileNotFoundError:
        print("[!] Warning: 'config.yaml' file not found. Ensure it is located in the same directory as this script.")
