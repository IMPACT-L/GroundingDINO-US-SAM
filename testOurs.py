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

csvPath = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/test.CSV'

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
#%% build SAM2 image predictor
SAM2_CHECKPOINT = '/home/hamze/Documents/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt'
SAM2_MODEL_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
sam2_model.eval()
sam2_predictor = SAM2ImagePredictor(sam2_model)
sam2_model.eval()
#%%
config_path="configs/test_config.yaml"
data_config, model_config, test_config = ConfigurationManager.load_config(config_path)
model = load_model(model_config,True)
model.eval()
#%%

is_unseen = True

if is_unseen:
    datasets = ["busbra","tnscui","luminous"]
else:
    datasets = ["breast", "buid", "busuc","busuclm","busb", "busi",
                "stu","s1","tn3k","tg3k","105us",
                "aul","muregpro","regpro","kidnyus"]
datasets = ["luminous"]
show_plots = True
margin = 0
box_threshold= 0.05
text_threshold=0.3
iou_threshold=10
threshold = .5

for selectedDataset in datasets:
    print("*"*20,selectedDataset,"*"*20)
    save_result_path = f'visualizations/ours/{selectedDataset}'
    os.makedirs(save_result_path, exist_ok=True)
    textCSV = getTextSample(selectedDataset)


    ious = []
    dices = []
    not_detected_list = []
    for image_index,image_name in enumerate(textCSV):
        caption = preprocess_caption(caption=textCSV[image_name]['promt_text'])
        image_path=os.path.join(data_config.val_dir,image_name)
        
        image_source, image = load_image(image_path)
        h, w, _ = image_source.shape
        boxes, _logits, phrases = predict(model,
                image,
                caption,
                box_threshold,
                text_threshold,
                remove_combined= True)


        iou = None
        dic = None

        detected = False
        if len(boxes>0):
            # print('brfore',_logits)
            boxes, _logits, phrases = apply_nms_per_phrase(image_source, boxes, _logits, phrases, box_threshold)
            best_box=_logits.argmax()
            # print('after',_logits,'\tbest_box',best_box)
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
            
            masks=masks[0]
            x1, y1, x2, y2 = rec1
            
            x1-=margin
            y1-=margin
            box_w = x2 - x1+2*margin
            box_h = y2 - y1+2*margin

            mask_path = os.path.join(data_config.val_dir.replace('test_image','test_mask'),image_name)
            mask_source = Image.open(mask_path).convert('L').resize((w,h))

            mask_source = np.asarray(mask_source).copy()
            mask_source[mask_source>=threshold]=1
            mask_source[mask_source<threshold]=0

            iou = sklearn_iou(masks,mask_source)*100
            dic = sklearn_dice(masks,mask_source)*100

            if show_plots:
                fig, ax = plt.subplots(1, 3, figsize=(20, 8))


                ax[0].set_title(f'Source: {image_name}')
                ax[0].axis('off')
                ax[0].imshow(image_source)


                tmp_image = image_source.copy()
                tmp_image[:,:,2][mask_source==1]=255
                ax[1].set_title(f'Ground Truth[{image_index}]')
                ax[1].axis('off')
                ax[1].imshow(tmp_image)
                
                tmp_image = image_source.copy()
                tmp_image[:,:,2][masks==1]=255
                ax[2].imshow(tmp_image)
                rect1 = patches.Rectangle((x1, y1), box_w, box_h,
                                        linewidth=2, edgecolor='red', facecolor='none')
                # ax[2].add_patch(rect1)
                ax[2].set_title(f'iou: {iou:.2f}, dice: {dic:.2f}')
                # plt.text(x=x1, y=y1, s=phrases, color='red', fontsize=10)
                ax[2].axis('off')
                plt.savefig(f"{save_result_path}/{image_name.replace('.bmp','.png')}") 
                # plt.show()
                plt.close()

            ious.append(iou)
            dices.append(dic)
        else:
            print(f'[{image_name}{image_index}] NO BOX FOUNDED FOR ')  
            not_detected_list.append(image_name)
    ious = np.array(ious)
    dices = np.array(dices)

    print(f"Number of not detected: {len(not_detected_list)}")
    print(f"Average IoU: {ious.mean():.2f}±{ious.std():.2f}")
    print(f"Average Dic: {dices.mean():.2f}±{dices.std():.2f}")

    with open(f'{save_result_path}/result.txt', 'w') as f:
        f.write(f"Average Dice, IoU: {dices.mean():.2f}±{dices.std():.0f} & {ious.mean():.2f}±{ious.std():.0f}\n")
    print(not_detected_list)
print('Finished')
# %%
