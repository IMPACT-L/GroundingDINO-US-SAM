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

#%%
desDir = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/USDATASET/test_annotation.CSV'

def getTextSample():
    textCSV = {}
    with open(desDir, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
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
                    'mask_path': row['mask_path']
                }
    return textCSV

textCSV = getTextSample()
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

    print(f"The unique detected phrases are {set(phrases)}")

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
SAM2_CHECKPOINT = '/home/hamze/Documents/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt'
SAM2_MODEL_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
sam2_predictor = SAM2ImagePredictor(sam2_model)
#%%
config_path="configs/test_config.yaml"
data_config, model_config, test_config = ConfigurationManager.load_config(config_path)
model = load_model(model_config,test_config.use_lora)
#%%
box_threshold=0.10
text_threshold=0.30
results = {}
for img in os.listdir(data_config.val_dir):
    caption = preprocess_caption(caption=textCSV[img]['promt_text'])
    image_path=os.path.join(data_config.val_dir,img)
    
    image_source, image = load_image(image_path)
    h, w, _ = image_source.shape
    boxes, logits, phrases = predict(model,
            image,
            caption,
            box_threshold,
            text_threshold,
            remove_combined= True)

    iou = None
    dic = None
    detected = False
    if len(boxes>0):
        boxes, logits, phrases = apply_nms_per_phrase(image_source, boxes, logits, phrases, box_threshold)

        sam2_predictor.set_image(np.array(image_source))
        rec = boxes * torch.tensor([w, h, w, h])
        rec = rec[0].cpu().numpy()
        rec1 = box_cxcywh_to_xyxy(rec)
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=rec1,
            multimask_output=False,
        )

        x1, y1, x2, y2 = rec1
        box_w = x2 - x1
        box_h = y2 - y1

        # Plot image and rectangle
        # Overlay the mask with transparency
        mask_path = textCSV[img]['mask_path']
        mask_source = Image.open(mask_path)
        mask_source = np.asarray(mask_source)
        iou = sklearn_iou(masks,mask_source)*100
        dic = sklearn_dice(masks,mask_source)*100

        fig, ax = plt.subplots(1, 3, figsize=(12, 8))


        ax[0].set_title(f'Source: {img}')
        ax[0].axis('off')
        ax[0].imshow(image_source)


        ax[1].set_title('Ground Truth')
        ax[1].axis('off')
        ax[1].imshow(mask_source)
        

        ax[2].imshow(image_source)

        # Draw the rectangle
        rect = patches.Rectangle((x1, y1), box_w, box_h,
                                linewidth=2, edgecolor='red', facecolor='none')
        ax[2].add_patch(rect)

        
        ax[2].imshow(masks[0], alpha=0.5)

        # ax.imshow(masks[0], alpha=0.5)

        ax[2].set_title(f'Prediction: iou: {iou:.2f}, dice: {dic:.2f}')
        plt.text(x=x1, y=y1, s=phrases, color='red', fontsize=10)
        ax[2].axis('off')
        plt.show()
        detected = True
    else:
        print('NO BOX FOUNDED FOR ',img)  
    results[img] ={
        'iou':iou,
        'dic':dic,
        'detected':detected
    }
    # break
# %%
undetected_results = {
    img: metrics for img, metrics in results.items() if not metrics['detected']
}

# Collect IOU and DICE values only from detected results
ious = [metrics['iou'] for metrics in results.values() if metrics['detected']]
dices = [metrics['dic'] for metrics in results.values() if metrics['detected']]

# Calculate average IOU and DICE, with zero fallback
avg_iou = sum(ious) / len(ious) if ious else 0.0
avg_dice = sum(dices) / len(dices) if dices else 0.0

# Output
print(f"Average IOU: {avg_iou:.4f}")
print(f"Average DICE: {avg_dice:.4f}")
print(f"Undetected Objects: {len(undetected_results)}")
# %%
