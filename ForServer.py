#%%
from groundingdino.util.inference import predict, load_model, load_image, predict
import torch
import torchvision.ops as ops
import os
from torchvision.ops import box_convert
import os
import torch
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from types import SimpleNamespace
import time
#%% Input Parameters
SAM2_CHECKPOINT = '/home/hamze/Documents/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt'
SAM2_MODEL_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
DEVICE = 'cpu' #'cuda'
image_path = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/test_image/luminous_7_4_bmode.tif'

model_config = SimpleNamespace()
model_config.config_path = 'groundingdino/config/GroundingDINO_SwinT_OGC.py'
model_config.lora_weigths = 'weights/20250608_1606/best_model.pth'
model_config.weights_path = 'weights/groundingdino_swint_ogc.pth'

top_k=3
box_threshold=0.01
text_threshold=0.02
text_prompt="left paraspinal muscle seen in the left side of the image"

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
sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
sam2_model.eval()
sam2_predictor = SAM2ImagePredictor(sam2_model)
#%%
model = load_model(model_config,True)
model.eval()
#%%
caption = preprocess_caption(caption=text_prompt)
image_source, image = load_image(image_path)
start_time = time.time()

h, w, _ = image_source.shape
boxes, logits, phrases = predict(model,
        image,
        caption,
        box_threshold,
        text_threshold,
        remove_combined= True)

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
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))

        ax[0].set_title(f'Source Image')
        ax[0].axis('off')
        ax[0].imshow(image_source)

        ax[1].imshow(image_source)
        ax[1].axis('off')
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

            overlay_mask[:,:,2][masks>0]=255
            x1, y1, x2, y2 = box_np[0]
            box_w = x2 - x1
            box_h = y2 - y1
        
            rect = patches.Rectangle((x1, y1), box_w, box_h,
                                        linewidth=2, edgecolor='red', facecolor='none')
            ax[1].add_patch(rect)

        ax[1].imshow(overlay_mask, cmap='gray')
        plt.show()
else:
    print('NO BOX FOUNDED')  

end_time = time.time()
# times.append(end_time - start_time)
print(f"Execution time: {end_time - start_time:.4f} seconds")
# %%
