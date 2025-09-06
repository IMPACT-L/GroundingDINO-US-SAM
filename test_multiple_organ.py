#%%
from groundingdino.util.inference import predict, load_model, load_image, predict
import torch
import torchvision.ops as ops
from torchvision.ops import box_convert
from groundingdino.util.inference import GroundingDINOVisualizer
from config import ConfigurationManager, DataConfig, ModelConfig
import torch
import numpy as np
from pathlib import Path
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
parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
parser.add_argument("--image_path", "-i", type=str, required=False, help="path to iamge",default='../Dataset/1-BreastDataset/Breast_BUS_B_2024/BUS/original/000001.png')
parser.add_argument("--mask_path", "-m", type=str, required=False, help="path to mask",default='../Dataset/1-BreastDataset/Breast_BUS_B_2024/BUS/GT/000001.png')
parser.add_argument("--text_prompt", "-t", type=str, required=False, help="text prompt",default='lumbar multifidus. benign cyst. benign. malignant. pants. text.')
parser.add_argument("--top_k", "-k", type=int, required=False, help="top_k",default=3)
parser.add_argument("--text_threshold", "-tt", type=float, required=False, help="text threshold",default=0.01)
parser.add_argument("--box_threshold", "-bt", type=float, required=False, help="box threshold",default=0.01)
parser.add_argument("--sam2_checkpoint", "-sc", type=str, required=False, help="sam2 checkpoint",default='../Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt')
parser.add_argument("--sam2_model_cfg", "-smc", type=str, required=False, help="sam2 model config",default='configs/sam2.1/sam2.1_hiera_l.yaml')
parser.add_argument("--model_config_path", "-mc", type=str, required=False, help="model config path",default='configs/test_config.yaml')




args = parser.parse_args()

box_threshold= args.box_threshold
text_threshold= args.text_threshold
top_k= args.top_k
image_path = args.image_path
mask_path = args.mask_path
text_prompt = args.text_prompt
sam2_checkpoint = args.sam2_checkpoint
model_cfg = args.sam2_model_cfg
config_path = args.model_config_path
caption = preprocess_caption(caption=text_prompt)
image_source, image = load_image(image_path)


#%% build SAM2 image predictor
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device='cuda')
sam2_model.eval()
sam2_predictor = SAM2ImagePredictor(sam2_model)
#%%
data_config, model_config, test_config = ConfigurationManager.load_config(config_path)
model = load_model(model_config,test_config.use_lora)
model.eval()
#%%
import time
start_time = time.time()

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
        
        ax[2].axis('off')
        overlay_mask = image_source.copy()
        
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
            ax[2].set_title(f'Pred IoU:{iou:.2f},DSC:{dic:.2f}')
            if dic>80 or dic < 10:
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

end_time = time.time()
# times.append(end_time - start_time)
print(f"Execution time: {end_time - start_time:.4f} seconds")
plt.imshow(overlay_mask, cmap='gray')
plt.axis('off')
# %%
