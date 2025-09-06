#%%
from groundingdino.util.inference import predict, load_model, load_image, predict, annotate
import torch
import torchvision.ops as ops
import os
from torchvision.ops import box_convert
from groundingdino.util.inference import GroundingDINOVisualizer
from config import ConfigurationManager, DataConfig, ModelConfig
import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from supervision.draw.color import ColorPalette
from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import jaccard_score, f1_score
import csv
import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#%%

def sklearn_iou(pred_mask, true_mask):
    return jaccard_score(true_mask.flatten(), pred_mask.flatten())

def sklearn_dice(pred_mask, true_mask):
    return f1_score(true_mask.flatten(), pred_mask.flatten())

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x#.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b

def apply_nms_per_phrase(image_source, boxes, logits, phrases, threshold=0.3):
    h, w, _ = image_source.shape
    scaled_boxes = boxes * torch.Tensor([w, h, w, h])
    scaled_boxes = box_convert(boxes=scaled_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    nms_boxes_list, nms_logits_list, nms_phrases_list = [], [], []

    # print(f"The unique detected phrases are {set(phrases)}")

    for unique_phrase in set(phrases):
        indices = [i for i, phrase in enumerate(phrases) if phrase == unique_phrase]
        phrase_scaled_boxes = scaled_boxes[indices]
        phrase_boxes = boxes[indices]
        phrase_logits = logits[indices]

        keep_indices = ops.nms(phrase_scaled_boxes, phrase_logits, threshold)
        nms_boxes_list.extend(phrase_boxes[keep_indices])
        nms_logits_list.extend(phrase_logits[keep_indices])
        nms_phrases_list.extend([unique_phrase] * len(keep_indices))

    return torch.stack(nms_boxes_list), torch.stack(nms_logits_list), nms_phrases_list

#%% build SAM2 image predictor
SAM2_CHECKPOINT = '../Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt'
SAM2_MODEL_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
sam2_model.eval()

sam2_predictor = SAM2ImagePredictor(sam2_model)
#%%
config_path="configs/test_config.yaml"
data_config, model_config, test_config = ConfigurationManager.load_config(config_path)
model = load_model(model_config,True)
model.eval()
#%%
csvPath = 'multimodal-data/test.CSV'
def getTextSample(dataset=None):
    textCSV = {}
    with open(csvPath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if dataset == None or row['dataset'] == dataset:
                textCSV[str(row['image_name'])] =\
                        {
                        'promt_text': row['label_name'],
                        'bbox': [
                            int(row['bbox_x']),
                            int(row['bbox_y']),
                            int(row['bbox_width']),
                            int(row['bbox_height'])
                        ],
                        'image_size': [
                            int(row['image_width']),
                            int(row['image_height'])
                        ],
                        # 'mask_path': row['mask_path']
                    }
    return textCSV

#%%
#%%
# Breast: busuclm,stu,s1, busi,busuc,busb,buid,breast,
# Liver: 105us, aul, 
# Kidny: kidnyus, 
# Prostate: muregpro,regpro,
# Thyroid: tg3k,tn3k,

#Unseen luminous, tnscui, busbra

box_threshold = 0.01
text_threshold = 0.01
threshold = 0.5

# Define target size for all images
TARGET_HEIGHT = 256
TARGET_WIDTH = 256
label_fontsize = 20

is_unseen = False

if is_unseen:
    dataTuple = [
        ('busbra', 'busbra_bus_0054-s.png'),
        ('luminous', 'luminous_106_24_bmode.tif'),
        ('tnscui', 'tnscui_175.png'),
    ]
else:
    dataTuple = [
        ('tg3k', 'tg3k_3546.jpg'), #tg3k_1488, tg3k_1322
        ('aul', 'aul_40.jpg'),
        ('kidnyus', 'kidnyus_625_im-0223-0256_anon_capsule.png'),
        ('breast', 'breast_case057.png'),
        ('muregpro', 'muregpro_5_t_9.png') 
    ]

# Create figure with adjusted subplot sizes
if is_unseen:
    fig, axes = plt.subplots(len(dataTuple), 8, figsize=(21, 9),
                         gridspec_kw={'width_ratios': [0.1, 1, 1, 1, 1, 1, 1, 1]})
else:
    fig, axes = plt.subplots(len(dataTuple), 8, figsize=(21, 15),
                         gridspec_kw={'width_ratios': [0.1, 1, 1, 1, 1, 1, 1, 1]})

plt.subplots_adjust(wspace=0.05, hspace=0.1)  # Adjust spacing between subplots

def resize_image(img, target_size=(TARGET_WIDTH, TARGET_HEIGHT)):
    """Resize image to target size while maintaining aspect ratio with padding"""
    if len(img.shape) == 2:  # Grayscale mask
        img = np.stack((img,)*3, axis=-1)
    
    # Resize maintaining aspect ratio
    h, w = img.shape[:2]
    scale = min(target_size[0]/w, target_size[1]/h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Pad to target size
    delta_w = target_size[0] - new_w
    delta_h = target_size[1] - new_h
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    
    color = [0, 0, 0]  # Black padding
    return cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

for row_idx, (selectedDataset, image_name) in enumerate(dataTuple):
    textCSV = getTextSample(selectedDataset)
    save_result_path = f'visualizations/SampleSegmentationResult/unseen/{selectedDataset}'\
        if is_unseen else f'visualizations/SampleSegmentationResult/seen/{selectedDataset}'
    os.makedirs(save_result_path, exist_ok=True)

    # ---- Load and process images ----
    image_path = os.path.join(data_config.val_dir, image_name)
    image_source, image = load_image(image_path)
    h, w, _ = image_source.shape
    
    # Load GT mask
    mask_path = os.path.join(data_config.val_dir.replace('test_image','test_mask'), image_name)
    mask_source = Image.open(mask_path).convert('L')
    mask_source = np.asarray(mask_source).copy()
    mask_source[mask_source >= threshold] = 1
    mask_source[mask_source < threshold] = 0

    # ---- Process predictions ----
    with torch.no_grad():
        boxes, _logits, phrases = predict(model, image, preprocess_caption(textCSV[image_name]['promt_text']), 
                                      box_threshold, text_threshold)
        boxes, _logits, phrases = apply_nms_per_phrase(image_source, boxes, _logits, phrases, box_threshold)
        best_box = _logits.argmax()

        sam2_predictor.set_image(np.array(image_source))
        rec = boxes * torch.tensor([w, h, w, h])
        rec = rec[best_box].cpu().numpy()
        rec1 = box_cxcywh_to_xyxy(rec)

        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=rec1,
            multimask_output=False,
        )
    masks = masks[0]
    masks[masks >= threshold] = 1
    masks[masks < threshold] = 0

    # Load MedClipSAM prediction
    if is_unseen:
        sam_clip_path = f'visualizations/GroundedSAM-US_unseen/MedCLIP-SAM/{selectedDataset}_unseen/{image_name}'.replace('png','npz').replace('jpg','npz').replace('.bmp','.npz').replace('.tif','.npz')
    else:
        sam_clip_path = f'multimodal-data/MedClipSamResults/MedCLIP-SAM/{selectedDataset}/{image_name}'.replace('png','npz').replace('jpg','npz').replace('.bmp','.npz')
    data = np.load(sam_clip_path)
    med_clip_sam = data['arr']
    med_clip_sam[med_clip_sam >= threshold] = 1
    med_clip_sam[med_clip_sam < threshold] = 0
    if med_clip_sam.shape[:2] != image_source.shape[:2]:
        med_clip_sam = cv2.resize(med_clip_sam.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    med_clip_sam[med_clip_sam >= threshold] = 1
    med_clip_sam[med_clip_sam < threshold] = 0

    # Load MedClipSAMv2 prediction
    if is_unseen:
        sam_clip_v2_path = f'visualizations/GroundedSAM-US_unseen/MedCLIP-SAMv2/{selectedDataset}_unseen/{image_name}'.replace('png','npz').replace('jpg','npz').replace('.bmp','.npz').replace('.tif','.npz')
    else:
        sam_clip_v2_path = f'multimodal-data/MedClipSamResults/MedCLIP-SAMv2/{selectedDataset}/{image_name}'.replace('png','npz').replace('jpg','npz').replace('.bmp','.npz')

    data = np.load(sam_clip_v2_path)
    med_clip_sam_v2 = data['arr']
    med_clip_sam_v2[med_clip_sam_v2 >= threshold] = 1
    med_clip_sam_v2[med_clip_sam_v2 < threshold] = 0

    # Load UniverSeg prediction
    univer_Seg_path = f'multimodal-data/UniverSeg/{selectedDataset}/{image_name}'.replace('png','npz').replace('jpg','npz').replace('.bmp','.npz').replace('.tif','.npz')
    
    data = np.load(univer_Seg_path)
    univer_seg_mask = data['arr']
    univer_seg_mask[univer_seg_mask >= threshold] = 1
    univer_seg_mask[univer_seg_mask < threshold] = 0
    if univer_seg_mask.shape[:2] != image_source.shape[:2]:
        univer_seg_mask = cv2.resize(univer_seg_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    univer_seg_mask[univer_seg_mask >= threshold] = 1
    univer_seg_mask[univer_seg_mask < threshold] = 0

    # Load BiomedParse prediction
    biomed_parse_path = f'multimodal-data/BiomedParse/{selectedDataset}/{image_name}'.replace('png','npz').replace('jpg','npz').replace('.bmp','.npz').replace('.tif','.npz')
    
    data = np.load(biomed_parse_path)
    biomed_parse_mask = data['logits']
    biomed_parse_mask[biomed_parse_mask >= threshold] = 1
    biomed_parse_mask[biomed_parse_mask < threshold] = 0
    if biomed_parse_mask.shape[:2] != image_source.shape[:2]:
        biomed_parse_mask = cv2.resize(biomed_parse_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    biomed_parse_mask[biomed_parse_mask >= threshold] = 1
    biomed_parse_mask[biomed_parse_mask < threshold] = 0

    # Load SAMUS prediction
    samus_path = f'multimodal-data/SAMUS/{selectedDataset}/{image_name}'.replace('png','npz').replace('jpg','npz').replace('.bmp','.npz').replace('.tif','.npz')
    
    data = np.load(samus_path)
    samus_mask = data['prediction']
    samus_mask[samus_mask >= threshold] = 1
    samus_mask[samus_mask < threshold] = 0
    if samus_mask.shape[:2] != image_source.shape[:2]:
        samus_mask = cv2.resize(samus_mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    samus_mask[samus_mask >= threshold] = 1
    samus_mask[samus_mask < threshold] = 0

    # ---- Create all 6 visualization images ----
    visualization_images = []
    
    # 1. Original image (resized)
    visualization_images.append(resize_image(image_source))

    # 2. UniverSeg (resized)
    univer_vis = image_source.copy()
    univer_vis[:, :, 2][univer_seg_mask == 1] = 255
    visualization_images.append(resize_image(univer_vis))

    # 3. BiomedParse (resized)
    biomed_parse_vis = image_source.copy()
    biomed_parse_vis[:, :, 2][biomed_parse_mask == 1] = 255
    visualization_images.append(resize_image(biomed_parse_vis))

    # 4. SAMUS (resized)
    samus_vis = image_source.copy()
    samus_vis[:, :, 2][samus_mask == 1] = 255
    visualization_images.append(resize_image(samus_vis))

    # 5. MedClipSAM (resized)
    medclip_vis = image_source.copy()
    medclip_vis[:, :, 2][med_clip_sam == 1] = 255
    visualization_images.append(resize_image(medclip_vis))
    
    # 6. MedClipSAMv2 (resized)
    medclipv2_vis = image_source.copy()
    medclipv2_vis[:, :, 2][med_clip_sam_v2 == 1] = 255
    visualization_images.append(resize_image(medclipv2_vis))

    # 7. Our method (resized)
    ours_vis = image_source.copy()
    ours_vis[:, :, 2][masks == 1] = 255
    visualization_images.append(resize_image(ours_vis))

    # 8. Ground Truth (resized)
    gt_vis = image_source.copy()
    gt_vis[:, :, 2][mask_source == 1] = 255
    visualization_images.append(resize_image(gt_vis))

    # ---- Plotting ----
    for col_idx in range(len(visualization_images)):
        ax = axes[row_idx, col_idx]
        ax.imshow(visualization_images[col_idx])
        
        if col_idx > 0 and col_idx < 7:  # Skip original image
            # Get the appropriate prediction mask
            pred_mask = None
            if col_idx == 1:  # UniverSeg
                pred_mask = univer_seg_mask
            if col_idx == 2:  # BiomedParse
                pred_mask = biomed_parse_mask
            elif col_idx == 3:  # SAMUS
                pred_mask = samus_mask    
            elif col_idx == 4:  # MedClipSAM
                pred_mask = med_clip_sam
            elif col_idx == 5:  # MedClipSAMv2
                pred_mask = med_clip_sam_v2            
            elif col_idx == 6:  # Our method
                pred_mask = masks

            
            
            # Calculate metrics
            iou = sklearn_iou(mask_source, pred_mask)*100
            dice = sklearn_dice(mask_source, pred_mask)*100
            
            # Display metrics on image
            text_str = f'IoU: {iou:.2f}\nDSC: {dice:.2f}'
            ax.text(0.5, 0.05, text_str, 
                transform=ax.transAxes,
                fontsize=label_fontsize, 
                color='white',
                ha='center',  # Center horizontally
                va='bottom',  # Align to bottom
                bbox=dict(facecolor='black', alpha=0.7, pad=4, edgecolor='none')
                )
            
        
        if row_idx == 0:
            titles = ["Original", "UniverSeg", "BiomedParse", "SAMUS", "MedClip-SAM", "MedClip-SAMv2", "Ours", "Ground Truth"]
            ax.set_title(titles[col_idx], fontsize=label_fontsize, pad=10, fontweight='bold')
            
        ax.set_xticks([])
        ax.set_yticks([])
        
        # First column - vertical dataset labels
        if col_idx == 0:
            # Remove the default subplot
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            # Add vertical text
            ax.text(0.5, 0.5, selectedDataset.capitalize(), 
                   rotation=90, va='center', ha='center',
                   fontsize=label_fontsize, fontweight='bold')
            
plt.tight_layout()
plt.savefig(f"visualizations/SampleSegmentationResult/{'un_seen' if is_unseen else 'seen'}.png", 
           dpi=300, bbox_inches='tight')
plt.show()