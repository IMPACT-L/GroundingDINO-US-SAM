#%%
from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
import torchvision.ops as ops
import os
from typing import Tuple, List
from torchvision import transforms as T
from torchvision.ops import box_convert
from groundingdino.util.inference import GroundingDINOVisualizer
from config import ConfigurationManager, DataConfig, ModelConfig
import shutil
import cv2
from groundingdino.datasets.dataset import GroundingDINODataset
from groundingdino.util.losses import SetCriterion
from torch.utils.data import DataLoader
from collections import defaultdict
from groundingdino.util.matchers import build_matcher
from groundingdino.util.misc import nested_tensor_from_tensor_list
import bisect
from groundingdino.util.utils import get_phrases_from_posmap
from matplotlib import pyplot as plt
import matplotlib.patches as patches

import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import torch
from segment_anything import sam_model_registry
from PIL import Image
import argparse
import csv
from sklearn.metrics import jaccard_score, f1_score

#%%
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

def show_box(box, ax, label=None,isBottom=False, color_index=0):
    color = np.array(color_palette[color_index % len(color_palette)])
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=2)
    )
    if label is not None:
        ax.text(x0, y0+ (h if isBottom else 0), label, color='white', fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.9, edgecolor='none', pad=1))


def show_mask(mask, ax, color_index=0):

    color = np.array(color_palette[color_index % len(color_palette)])

    h, w = mask.shape[-2:]

    if mask.max() > 1:
        mask = mask / 255.0

    # Create RGBA image from mask and color
    mask_image = np.zeros((h, w, 4))
    for i in range(4):
        mask_image[:, :, i] = mask * color[i]

    ax.imshow(mask_image)

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def sklearn_iou(pred_mask, true_mask):
    return jaccard_score(true_mask.flatten(), pred_mask.flatten())

def sklearn_dice(pred_mask, true_mask):
    return f1_score(true_mask.flatten(), pred_mask.flatten())

def load_image_medsam(image_path: str)-> Tuple[np.array, torch.Tensor]:
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
#%%
config_path="configs/test_config.yaml"

data_config, model_config, test_config = ConfigurationManager.load_config(config_path)
model = load_model(model_config,test_config.use_lora)

device = 'cuda'
model = model.to(device)
model.eval()

medsam_model = sam_model_registry["vit_b"](checkpoint="/home/hamze/Documents/MedSAM/work_dir/MedSAM/medsam_vit_b.pth")
medsam_model = medsam_model.to(device)
medsam_model.eval()
#%%
terminal = False
top_k=3
box_threshold=0.1
text_threshold=0.1
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
# image_path = '/home/hamze/Documents/Dataset/BreastBUSI_Images/malignant/malignant (140).png'
# mask_path = '/home/hamze/Documents/Dataset/BreastBUSI_Images/malignant/malignant (140)_mask.png'
image_path = '/home/hamze/Documents/Dataset/CCAUI/Image/202201121748100022VAS_slice_1069.png'
mask_path = '//home/hamze/Documents/Dataset/CCAUI/Mask/202201121748100022VAS_slice_1069.png'
# image_path = 'sample_tests/two_dogs.png'
# mask_path = 'sample_tests/two_dogs.png'
# image_path = '/home/hamze/Documents/Dataset/BUSBRA/Images/bus_0064-s.png'
# image_path = '/home/hamze/Documents/Dataset/BUS-UCLM Breast ultrasound lesion segmentation dataset/images/ALWI_000.png'
# image_path = 'samples_with_text.png'
#** image_path = '/home/hamze/Documents/Dataset/fetal head circumference/training_set/001_HC.png'
#** image_path = '/home/hamze/Documents/Dataset/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks/case001.png'
# image_path = '/home/hamze/Documents/Dataset/kidneyUS_images_14_june_2022/kidneyUS_images_14_june_2022/1_IM-0001-0059_anon.png'
# image_path = '/home/hamze/Documents/Dataset/LUMINOUS_Database/B-mode/54_1_Bmode.tif'
# image_path = '/home/hamze/Documents/Dataset/Thyroid Dataset/DDTI dataset/DDTI/1_or_data/image/3.PNG'
# image_path = '/home/hamze/Documents/Dataset/Thyroid Dataset/tg3k/thyroid-image/0000.jpg'
# text_prompt="thyroid. lumbar multifidus. benign cyst. benign. malignant. pants. text." #1
# text_prompt="find malignant on the center of the image." #1
# text_prompt= "chair . person . dog ."
text_prompt="benign . malignant . chair . person . dog ." #1
if terminal and args.path:
    image_path =  args.path


if terminal and args.text_prompt:
    text_prompt = args.text_prompt

caption = preprocess_caption(caption=text_prompt)
image_source, image = load_image_medsam(image_path)
h, w, _ = image_source.shape
boxes, logits, phrases  = predict(model,
        image,
        caption,
        box_threshold,
        text_threshold,
        remove_combined= False)

iou = None
dic = None

if len(boxes>0):

    boxes, logits, phrases = apply_nms_per_phrase(image_source, boxes, logits, phrases,box_threshold)
    # _, top_indices = torch.topk(logits, top_k if boxes.shape[0]>=top_k else boxes.shape[0])

    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(image.float().permute(0, 2, 1).unsqueeze(0).to(device))
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
            box_1024 = box_np / np.array([w,h,w,h]) * 1024
            masks = medsam_inference(medsam_model, image_embedding, box_1024, h,w)
            iou = sklearn_iou(masks,mask_source)*100
            dic = sklearn_dice(masks,mask_source)*100
            if iou>=0:
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