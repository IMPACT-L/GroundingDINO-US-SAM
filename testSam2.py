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

#%%
def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x#.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return b


#%% build SAM2 image predictor
SAM2_CHECKPOINT = '/home/hamze/Documents/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt'
SAM2_MODEL_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
sam2_predictor = SAM2ImagePredictor(sam2_model)
#%%

# Config file of the prediction, the model weights can be complete model weights but if use_lora is true then lora_wights should also be present see example
## config file
config_path="configs/test_config.yaml"
# text_prompt="shirt .bag .pants"
text_prompt="a malignant . a benign ."
# text_prompt="shirt .bag .pants"
data_config, model_config, training_config = ConfigurationManager.load_config(config_path)
model = load_model(model_config,training_config.use_lora)
#%%
box_threshold=0.30
text_threshold=0.20
caption = preprocess_caption(caption=text_prompt)

for img in os.listdir(data_config.val_dir):
    image_path=os.path.join(data_config.val_dir,img)
    #     image_source = Image.open(image_path).convert("RGB")
    image_source, image = load_image(image_path)
    h, w, _ = image_source.shape
    pre = predict(model,
            image,
            caption,
            box_threshold,
            text_threshold,
            remove_combined= True)
    print(img,pre)

    if len(pre[0]>0):
        sam2_predictor.set_image(np.array(image_source))
        rec = pre[0] * torch.tensor([w, h, w, h])
        rec = rec[0].cpu().numpy()
        rec1 = box_cxcywh_to_xyxy(rec)
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=rec1,
            multimask_output=True,
        )

        x1, y1, x2, y2 = rec1
        box_w = x2 - x1
        box_h = y2 - y1

        # Plot image and rectangle
        fig, ax = plt.subplots()
        ax.imshow(image_source)

        # Draw the rectangle
        rect = patches.Rectangle((x1, y1), box_w, box_h,
                                linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        # Overlay the mask with transparency
        ax.imshow(masks[0], alpha=0.5)

        # ax.imshow(masks[0], alpha=0.5)

        plt.title(img)
        plt.text(x=x1, y=y1, s=pre[2], color='red', fontsize=10)
        # plt.axis("off")
        plt.show()
#%%