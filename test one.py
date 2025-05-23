#%%
from groundingdino.util.inference import load_model, load_image, predict, annotate
import torch
import torchvision.ops as ops
import os
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
import torch.nn.functional as F
from skimage import io, transform
import numpy as np
import torch
from segment_anything import sam_model_registry

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

def show_box(box, ax, label=None,isBottom=False):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="white", facecolor=(0, 0, 0, 0), lw=2)
    )
    if label is not None:
        ax.text(x0, y0+ (h if isBottom else 0), label, color='white', fontsize=10, ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.9, edgecolor='none', pad=1))


def show_mask(mask, ax, color_index=0):

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

#%%
# if __name__ == "__main__":

text_threshold=0.1
prompt_type = 6
top_k=2

# Config file of the prediction, the model weights can be complete model weights but if use_lora is true then lora_wights should also be present see example
## config file
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
image_path = 'multimodal-data/Breast/images/train/000002.png'

image_path = '/home/hamze/Documents/Dataset/BULI_Malignant/112 Malignant Image.bmp'
image_source, image = load_image(image_path)

image = image.to(device)
# caption="benign cyst. benign. malignant. pants." #1
text_prompt="benign cyst. benign. malignant. pants." #1

with torch.no_grad():
    outputs = model(image[None], captions=[text_prompt])

prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

box_threshold=0.1
mask = prediction_logits.max(dim=1)[0] > box_threshold
logits = prediction_logits[mask]  # logits.shape = (n, 256)
boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)
remove_combined = True
tokenizer = model.tokenizer
tokenized = tokenizer(text_prompt)

if remove_combined:
    sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
    
    phrases = []
    for logit in logits:
        max_idx = logit.argmax()
        insert_idx = bisect.bisect_left(sep_idx, max_idx)
        right_idx = sep_idx[insert_idx]
        left_idx = sep_idx[insert_idx - 1]
        phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
else:
    phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
        for logit
        in logits
    ]
logits = logits.max(dim=1)[0]

if boxes.shape[0]>0:
    boxes, logits, phrases = apply_nms_per_phrase(image_source, boxes, logits, phrases)
    print(f"NMS boxes size {boxes.shape}")

top_k = 3
_, top_indices = torch.topk(logits, top_k if boxes.shape[0]>=top_k else boxes.shape[0])

annotated_frame = annotate(image_source=image_source, 
                            boxes=boxes[top_indices,:], 
                            logits=logits[top_indices], 
                            phrases=[phrases[i] for i in top_indices]
                            )

print([phrases[i] for i in top_indices],logits[top_indices])
# %%
h, w, _ = image_source.shape
boxes = boxes[top_indices,:] * torch.Tensor([w, h, w, h])

xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
rec1 = xyxy

phrases = [phrases[i] for i in top_indices]

combined = list(zip(rec1, phrases))
combined_sorted = sorted(
    combined,
    key=lambda x: (x[0][2] - x[0][0]) * (x[0][3] - x[0][1]),  # width * height
    reverse=True 
)

# Unzip back into separate lists
rec1_sorted, phrases_sorted = zip(*combined_sorted)
rec1_sorted = list(rec1_sorted)
phrases_sorted = list(phrases_sorted)


img_np = io.imread(image_path)
if len(img_np.shape) == 2:
    img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
else:
    img_3c = img_np
H, W, _ = img_3c.shape

img_1024 = transform.resize(
    img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
).astype(np.uint8)
img_1024 = (img_1024 - img_1024.min()) / np.clip(
    img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
)  # normalize to [0, 1], (H, W, 3)
# convert the shape to (3, H, W)
img_1024_tensor = (
    torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
)

medsam_segs=[]
for rc in rec1_sorted:
    box_np = np.array([[int(x) for x in rc]]) 
    # transfer box_np t0 1024x1024 scale
    box_1024 = box_np / np.array([W, H, W, H]) * 1024
    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)

    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
    medsam_segs.append(medsam_seg)

fig, ax = plt.subplots(1, 2, figsize=(16, 6))  # Wider figure for 3 columns

ax[0].imshow(annotated_frame)
ax[0].set_title([f'{phrases[i]}:{logits[i]:.3f}' for i in top_indices])
ax[0].axis('off')
# 1. Original image with masks and boxes
ax[1].imshow(img_3c)
for color_index, medsam_seg in enumerate(medsam_segs):
    show_mask(medsam_seg, ax[1], color_index=color_index)
    show_box(
        np.array([int(x) for x in rec1_sorted[color_index]]),
        ax[1],
        label=f"{phrases_sorted[color_index]}",
        isBottom=(color_index % 2 == 0)
    )
ax[1].set_title(f"MedSAM Segs ({len(medsam_segs)})")
ax[1].axis('off')
plt.tight_layout()
plt.show()

# plt.imshow(annotated_frame)
# plt.axis('off')
# plt.legend()
# plt.title([f'{phrases[i]}:{logits[i]:.3f}' for i in top_indices])
# plt.tight_layout()
# plt.show()
# %%
