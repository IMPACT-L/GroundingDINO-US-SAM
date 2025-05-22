#%%
# import argparse
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
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection 
#%%
# TEXT_PROMPT = 'car. tire. window. handle. light. Segment the dilated renal pelvis'
# VERY important: text queries need to be lowercased + end with a dot
TEXT_PROMPT = 'exhaust window and tires and excude tumor benign malignant lesion'
TEXT_PROMPT = TEXT_PROMPT.split()
TEXT_PROMPT = '. '.join(TEXT_PROMPT) + '.' 
# IMG_PATH = '/home/hamze/Documents/Dataset/BULI_Malignant/3 Malignant Image.bmp' 
# IMG_PATH =  'notebooks/images/truck.jpg'
# SAM2_CHECKPOINT = './checkpoints/sam2.1_hiera_large.pt'
# SAM2_MODEL_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
DEVICE = "cuda" #if torch.cuda.is_available() and not args.force_cpu else "cpu"
OUTPUT_DIR = Path('outputs/test_sam2.1')
DUMP_JSON_RESULTS = True #not args.no_dump_json
# %%create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
#%% build grounding dino from huggingface

# IMG_PATH = '/home/hamze/Documents/Dataset/BULI_Malignant/3 Malignant Image.bmp' #GOOD works
# IMG_PATH ='/home/hamze/Documents/Dataset/BreastBUSI_Images/benign/benign (1).png'
IMG_PATH ='/home/hamze/Documents/Dataset/BULI_Malignant/4 Malignant Image.bmp'
model_id = 'IDEA-Research/grounding-dino-tiny'
# model_id = 'IDEA-Research/grounding-dino-base'
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

image = Image.open(IMG_PATH)

inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    outputs = grounding_model(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs,
    inputs.input_ids,
    box_threshold=0.4,
    text_threshold=0.3,
    target_sizes=[image.size[::-1]]
)

print(results)
input_boxes = results[0]["boxes"].cpu().numpy()

if len(results)==0:
    print('Object Not Found')
else:
    print('Object Found')


#%% build SAM2 image predictor
SAM2_CHECKPOINT = '/home/hamze/Documents/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt'
SAM2_MODEL_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)
sam2_predictor.set_image(np.array(image.convert("RGB")))
masks, scores, logits = sam2_predictor.predict(
    point_coords=None,
    point_labels=None,
    box=input_boxes,
    multimask_output=False,
)

# %%
if masks.ndim == 4:
    masks = masks.squeeze(1)

confidences = results[0]["scores"].cpu().numpy().tolist()
class_names = results[0]["labels"]
class_ids = np.array(list(range(len(class_names))))

labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence
    in zip(class_names, confidences)
]
for i in range(0,len(confidences)):
    print(class_names[i],confidences[i])
#%%
"""
Visualize image with supervision useful API
"""
img = cv2.imread(IMG_PATH)
detections = sv.Detections(
    xyxy=input_boxes,  # (n, 4)
    mask=masks.astype(bool),  # (n, h, w)
    class_id=class_ids
)

"""
Note that if you want to use default color map,
you can set color=ColorPalette.DEFAULT
"""
box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
cv2.imwrite(os.path.join(OUTPUT_DIR, "groundingdino_annotated_image.jpg"), annotated_frame)

mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
cv2.imwrite(os.path.join(OUTPUT_DIR, "grounded_sam2_annotated_image_with_mask.jpg"), annotated_frame)


"""
Dump the results in standard format and save as json files
"""

def single_mask_to_rle(mask):
    rle = mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle

if DUMP_JSON_RESULTS:
    # convert mask into rle format
    mask_rles = [single_mask_to_rle(mask) for mask in masks]

    input_boxes = input_boxes.tolist()
    scores = scores.tolist()
    # save the results in standard format
    results = {
        "image_path": IMG_PATH,
        "annotations" : [
            {
                "class_name": class_name,
                "bbox": box,
                "segmentation": mask_rle,
                "score": score,
            }
            for class_name, box, mask_rle, score in zip(class_names, input_boxes, mask_rles, scores)
        ],
        "box_format": "xyxy",
        "img_width": image.width,
        "img_height": image.height,
    }
    
    with open(os.path.join(OUTPUT_DIR, "grounded_sam2_hf_model_demo_results.json"), "w") as f:
        json.dump(results, f, indent=4)

#%%
# Display final annotated image with matplotlib
plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.imshow(image)
plt.axis("off")
plt.title("Grounded SAM2 + Grounding DINO Results")
plt.subplot(122)
plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Grounded SAM2 + Grounding DINO Results")
plt.show()

# %%
