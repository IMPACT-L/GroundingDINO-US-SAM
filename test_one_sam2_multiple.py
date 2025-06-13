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
from sklearn.metrics import jaccard_score, f1_score
import argparse
#%%
desDir = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/USDATASET/test_annotation.CSV'

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
terminal = False
top_k=5
box_threshold=0.01
text_threshold=0.02
# python test_one.py -p /home/hamze/Documents/Dataset/LUMINOUS_Database/B-mode/54_27_Bmode.tif -t "lumbar_multifidus. text." -k 1 -tt 0.1 -bt .01

if terminal:
    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument("--path", "-p", type=str, required=False, help="path to iamge",default='/home/hamze/Documents/Dataset/LUMINOUS_Database/B-mode/54_1_Bmode.tif')
    parser.add_argument("--text_prompt", "-t", type=str, required=False, help="text prompt",default='lumbar multifidus. benign cyst. benign. malignant. pants. text.')
    parser.add_argument("--top_k", "-k", type=int, required=False, help="top_k",default=3)
    parser.add_argument("--text_threshold", "-tt", type=float, required=False, help="text threshold",default=0.01)
    parser.add_argument("--box_threshold", "-bt", type=float, required=False, help="box threshold",default=0.01)
    args = parser.parse_args()

if terminal and args.box_threshold:
    box_threshold= args.box_threshold
    

if terminal and args.text_threshold:
    text_threshold= args.text_threshold

if terminal and args.top_k:
    top_k= args.top_k
# image_path = 'multimodal-data/Breast/images/train/000002.png'
# image_path = '/home/hamze/Documents/Dataset/CCAUI/Image/202201121748100022VAS_slice_1069.png'
# mask_path = '//home/hamze/Documents/Dataset/CCAUI/Mask/202201121748100022VAS_slice_1069.png'

# image_path = '/home/hamze/Documents/Dataset/1-BreastDataset/BreastBUSI_Images/malignant/malignant (140).png'
# mask_path = '/home/hamze/Documents/Dataset/1-BreastDataset/BreastBUSI_Images/malignant/malignant (140)_mask.png'

# image_path = '/home/hamze/Documents/Dataset/2-Thyroid-Dataset/tg3k/thyroid-image/0805.jpg'
# mask_path = '/home/hamze/Documents/Dataset/2-Thyroid-Dataset/tg3k/thyroid-mask/0805.jpg'

# image_path = 'sample_tests/two_dogs.png'
# mask_path = 'sample_tests/two_dogs.png'
# image_path = '/home/hamze/Documents/Dataset/BUSBRA/Images/bus_0064-s.png'
# image_path = '/home/hamze/Documents/Dataset/BUS-UCLM Breast ultrasound lesion segmentation dataset/images/ALWI_000.png'
#** image_path = '/home/hamze/Documents/Dataset/fetal head circumference/training_set/001_HC.png'
#** image_path = '/home/hamze/Documents/Dataset/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/case001.png'
# image_path = '/home/hamze/Documents/Dataset/kidneyUS_images_14_june_2022/kidneyUS_images_14_june_2022/1_IM-0001-0059_anon.png'
# image_path = '/home/hamze/Documents/Dataset/LUMINOUS_Database/B-mode/54_1_Bmode.tif'
# image_path = '/home/hamze/Documents/Dataset/Thyroid Dataset/DDTI dataset/DDTI/1_or_data/image/3.PNG'
# image_path = '/home/hamze/Documents/Dataset/Thyroid Dataset/tg3k/thyroid-image/0000.jpg'
# text_prompt="thyroid. lumbar multifidus. benign cyst. benign. malignant. pants. text." #1
# text_prompt="find malignant on the center of the image." #1
# text_prompt= "chair . person . dog ."

image_path = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/test_image/busi_benign (34).png'
mask_path = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/test_mask/busi_benign (34).png'

image_path = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/test_image/busi_benign (83).png'
mask_path = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/test_mask/busi_benign (83).png'

image_path = '/home/hamze/Downloads/algorithms-16-00521-g001.png'
mask_path = '/home/hamze/Downloads/algorithms-16-00521-g001.png'

# text_prompt="carotid . benign . malignant . chair . person . dog ." #1
text_prompt="tumor. thyroid. carotid . benign . malignant . chair . person . dog ." #1
if terminal and args.path:
    image_path =  args.path


if terminal and args.text_prompt:
    text_prompt = args.text_prompt

caption = preprocess_caption(caption=text_prompt)
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
if len(boxes>0):

    boxes, logits, phrases = apply_nms_per_phrase(image_source, boxes, logits, phrases,box_threshold)
    _, top_indices = torch.topk(logits, top_k if boxes.shape[0]>=top_k else boxes.shape[0])
    boxes=boxes[top_indices]
    logits=logits[top_indices]
    phrases=[phrases[i] for i in top_indices]
    with torch.no_grad():
        
        sam2_predictor.set_image(np.array(image_source))
        boxes = boxes* torch.Tensor([w, h, w, h])

        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        fig, ax = plt.subplots(1, 3, figsize=(12, 8))
        ax[0].set_title(f'Source Image')
        ax[0].axis('off')
        ax[0].imshow(image_source)

        mask_source = Image.open(mask_path).convert('L')
        mask_source = np.asarray(mask_source).copy()
        mask_source[mask_source>0]=1

        ax[1].set_title(f'Ground Truth')
        ax[1].axis('off')
        ax[1].imshow(mask_source)
        
        # ax[2].imshow(image_source)
        ax[2].axis('off')
        overlay_mask = image_source.copy()
        # overlay_mask[:]=0
        ax[2].set_title('Prediction')
        for i , xyxy_ in enumerate(xyxy):
            box_np = np.array([[int(x) for x in xyxy_]]) 

            masks, _, _ = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box_np,
                multimask_output=False,
            )
            
            masks=masks[0]
            x1, y1, x2, y2 = box_np[0]

            iou = sklearn_iou(masks,mask_source)*100
            dic = sklearn_dice(masks,mask_source)*100
            if iou>=3:
                overlay_mask[:,:,2][masks>0]=255
                x1, y1, x2, y2 = box_np[0]
                box_w = x2 - x1
                box_h = y2 - y1
            
                rect = patches.Rectangle((x1, y1), box_w, box_h,
                                        linewidth=2, edgecolor='red', facecolor='none')
                ax[2].add_patch(rect)
            
            print(f'{iou:.2f}, dice: {dic:.2f}, phrase:{phrases[i]}, score:{logits[i]:.2f}')
            detected = True
        ax[2].imshow(overlay_mask, cmap='gray')
        plt.show()
else:
    print('NO BOX FOUNDED')  

# %%
