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
#%%
import csv
csvPath = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/test.CSV'
selectedDataset = None
selectedDataset =  'busuclm' #'tnscui'#'stu' #'breast' #'tn3k'#'tg3k'#'tnscui'
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
textCSV = getTextSample(selectedDataset)
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
show_plots = True
margin = 5
box_threshold=0.05
text_threshold=0.3
iou_threshold=10
cc_treshold=(15, 15)

ious_before = []
dices_before = []
ious_after = []
dices_after = []
dices_after = []
not_detected_list = []
for image_index,image_name in enumerate(textCSV):
    caption = preprocess_caption(caption=textCSV[image_name]['promt_text'])
    image_path=os.path.join(data_config.val_dir,image_name)
    
    image_source, image = load_image(image_path)
    h, w, _ = image_source.shape
    boxes, logits, phrases = predict(model,
            image,
            caption,
            box_threshold,
            text_threshold,
            remove_combined= True)


    iou_before = None
    dic_before = None
    iou_after = None
    dic_after = None
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
        
        masks=masks[0]
        x1, y1, x2, y2 = rec1
        
        x1-=margin
        y1-=margin
        box_w = x2 - x1+2*margin
        box_h = y2 - y1+2*margin

        mask_path = os.path.join(data_config.val_dir.replace('test_image','test_mask'),image_name)
        mask_source = Image.open(mask_path).convert('L').resize((w,h))
        mask_source
        mask_source = np.asarray(mask_source).copy()
        mask_source[mask_source>0]=1

        iou_before = sklearn_iou(masks,mask_source)*100
        dic_before = sklearn_dice(masks,mask_source)*100

        mask_uint8 = (masks * 255).astype(np.uint8)
        num_labels, labels_im = cv2.connectedComponents(mask_uint8)

        binary_mask = (mask_uint8 > 0).astype(np.uint8)

        # Fill holes
        inverted = cv2.bitwise_not(binary_mask * 255)
        h, w = inverted.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(inverted, mask, (0, 0), 255)
        inverted_filled = cv2.bitwise_not(inverted)
        filled_mask = binary_mask | (inverted_filled > 0).astype(np.uint8)

        # Connect components
        kernel = np.ones(cc_treshold, np.uint8)
        connected_mask = cv2.morphologyEx(filled_mask, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(connected_mask, kernel, iterations=1)
        iou_after = sklearn_iou(dilated,mask_source)*100
        dic_after = sklearn_dice(dilated,mask_source)*100

        if show_plots:
            fig, ax = plt.subplots(1, 4, figsize=(20, 8))


            ax[0].set_title(f'Source: {image_name}')
            ax[0].axis('off')
            ax[0].imshow(image_source)


            ax[1].set_title(f'Ground Truth[{image_index}]')
            ax[1].axis('off')
            ax[1].imshow(mask_source)
            
            tmp_image = image_source.copy()
            tmp_image[:,:,2][masks==1]=255
            ax[2].imshow(tmp_image)
            rect1 = patches.Rectangle((x1, y1), box_w, box_h,
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax[2].add_patch(rect1)
            ax[2].set_title(f'iou_before: {iou_before:.2f}, dice_before: {dic_before:.2f}')
            # plt.text(x=x1, y=y1, s=phrases, color='red', fontsize=10)
            ax[2].axis('off')

            # ax[3].imshow(image_source)
            tmp_image = image_source.copy()
            tmp_image[:,:,2][connected_mask==1]=255
            rect2 = patches.Rectangle((x1, y1), box_w, box_h,
                                    linewidth=2, edgecolor='red', facecolor='none')
            ax[3].imshow(tmp_image)
            ax[3].add_patch(rect2)
            ax[3].set_title(f'iou_after: {iou_after:.2f}, dice_after: {dic_after:.2f}')
            ax[3].axis('off')
            # ax[3].imshow(connected_mask, alpha=0.5)

            plt.show()
        if iou_after>iou_threshold:
            detected = True
        if detected:
            ious_before.append(iou_before)
            dices_before.append(dic_before)
            ious_after.append(iou_after)
            dices_after.append(dic_after)
        # else:
        #     not_detected_count += 1
    else:
        print(f'[{image_name}{image_index}] NO BOX FOUNDED FOR ')  
        not_detected_list.append(image_name)
    image_index += 1   
            
        # break
#%%
ious_before = np.array(ious_before)
dices_before = np.array(dices_before)
ious_after = np.array(ious_after)
dices_after = np.array(dices_after)

print(f"Number of not detected: {len(not_detected_list)}")
print(f"Average IoU: {ious_before.mean():.2f}±{ious_before.std():.2f} -> {ious_after.mean():.2f}±{ious_after.std():.2f}")
print(f"Average Dic: {dices_before.mean():.2f}±{dices_before.std():.2f}-> {dices_after.mean():.2f}±{dices_after.std():.2f}")
print(f"Min IoU[{1+ious_after.argmin()}]: {ious_after.min():.2f}")
print(f"Max IoU[{1+ious_after.argmax()}]: {ious_after.max():.2f}")

print(not_detected_list)
# %%
