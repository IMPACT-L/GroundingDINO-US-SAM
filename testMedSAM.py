
#%%
import torch
import torchvision
print(torch.cuda.is_available())  # Should return True if GPU is properly configured
# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms as T
from typing import Tuple, List

join = os.path.join
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
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
import warnings
warnings.filterwarnings("ignore")
import csv

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


def show_mask(mask, ax, color_index=0):
    import numpy as np

    # Define a list of RGBA colors
    color_palette = [
        [1.0, 0.0, 0.0, 0.6],  # Red
        [0.0, 1.0, 0.0, 0.6],  # Green
        [0.0, 0.0, 1.0, 0.6],  # Blue
        [1.0, 1.0, 0.0, 0.6],  # Yellow
        [1.0, 0.0, 1.0, 0.6],  # Magenta
        [0.0, 1.0, 1.0, 0.6],  # Cyan
        [1.0, 0.5, 0.0, 0.6],  # Orange
    ]

    color = np.array(color_palette[color_index % len(color_palette)])

    h, w = mask.shape[-2:]

    if mask.max() > 1:
        mask = mask / 255.0

    # Create RGBA image from mask and color
    mask_image = np.zeros((h, w, 4))
    for i in range(4):
        mask_image[:, :, i] = mask * color[i]

    ax.imshow(mask_image)

    


def show_box(box, ax, label=None,isBottom=False):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="white", facecolor=(0, 0, 0, 0), lw=2)
    )
    if label is not None:
        ax.text(x0, y0+ (h if isBottom else 0), label, color='white', fontsize=10, ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.9, edgecolor='none', pad=1))


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



@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

def load_image(image_path: str)-> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.Resize((1024, 1024)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed = transform(image_source)
    return image, image_transformed

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
device = 'cuda'
medsam_model = sam_model_registry["vit_b"](checkpoint="/home/hamze/Documents/MedSAM/work_dir/MedSAM/medsam_vit_b.pth")
medsam_model = medsam_model.to(device)
medsam_model.eval()
#%%
config_path="configs/test_config.yaml"
data_config, model_config, training_config = ConfigurationManager.load_config(config_path)
model = load_model(model_config,training_config.use_lora)
model.eval()
#%%
import csv
csvPath = 'multimodal-data/test.CSV'

is_unseen = True

if is_unseen:
    datasets = ["busbra","tnscui","luminous"]
else:
    datasets = ["breast", "buid", "busuc","busuclm","busb", "busi",
                "stu","s1","tn3k","tg3k","105us",
                "aul","muregpro","regpro","kidnyus"]

show_plots = True
margin = 0
box_threshold=0.05
text_threshold=0.3
threshold = .5
for selectedDataset in datasets:
    print("*"*20,selectedDataset,"*"*20)
    save_result_path = f'visualizations/MedSam/{selectedDataset}'
    os.makedirs(save_result_path, exist_ok=True)

    textCSV = getTextSample(selectedDataset)

    ious = []
    dices = []
    ious_after = []
    dices_after = []
    not_detected_list = []
    for image_index,image_name in enumerate(textCSV):
        caption = preprocess_caption(caption=textCSV[image_name]['promt_text'])

        image_path=os.path.join(data_config.val_dir,image_name)
        
        image_source, image = load_image(image_path)
        h, w, _ = image_source.shape
        boxes, _logits, phrases  = predict(model,
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
        
            with torch.no_grad():
                image_embedding = medsam_model.image_encoder(image.float().permute(0, 2, 1).unsqueeze(0).to(device))
            boxes = boxes* torch.Tensor([w, h, w, h])

            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            box_np = np.array([[int(x) for x in xyxy[0]]]) 
            box_1024 = box_np / np.array([w,h,w,h]) * 1024
            masks = medsam_inference(medsam_model, image_embedding, box_1024, h,w)

            x1, y1, x2, y2 = box_np[0]
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
                ax[2].add_patch(rect1)
                ax[2].set_title(f'iou: {iou:.2f}, dice: {dic:.2f}')
                ax[2].axis('off')

                plt.savefig(f"{save_result_path}/{image_name.replace('.bmp','.png')}") 
                plt.close()
            ious.append(iou)
            dices.append(dic)
        else:
            print(f'[{image_name}{image_index}]NO BOX FOUNDED FOR ')  
            not_detected_list.append(image_name)
    ious = np.array(ious)
    dices = np.array(dices)

    print(f"Number of not detected: {len(not_detected_list)}")
    print(f"Average IoU: {ious.mean():.2f}±{ious.std():.2f}")
    print(f"Average Dic: {dices.mean():.2f}±{dices.std():.2f}")

    with open(f'{save_result_path}/result.txt', 'w') as f:
        f.write(f"Average Dice, IoU: {dices.mean():.2f}±{dices.std():.0f} & {ious.mean():.2f}±{ious.std():.0f}\n")
print('Finished')
# %%
