
#%%
import torch
import torchvision
print(torch.cuda.is_available())  # Should return True if GPU is properly configured
# %% load environment
import numpy as np
import matplotlib.pyplot as plt
import os

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
import csv

#%%
desDir = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/USDATASET/test_annotation.CSV'

def getTextSample():
    textCSV = {}
    with open(desDir, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
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
                    'mask_path': row['mask_path']
                }
    return textCSV
textCSV = getTextSample()

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


#%% load model and image
box_threshold=0.01
text_threshold=0.01
prompt_type = 1
prompt_type = 'lumbar_multifidus'
top_k=1
save_path = f"visualizations/Pre/medsam_{prompt_type}_{box_threshold}_{text_threshold}"
# shutil.rmtree ("visualizations/inference")
os.makedirs(save_path, exist_ok=True)

args={
    "data_path":"/home/hamze/Documents/Dataset/BreastBUSI_Images/benign/benign (264).png",
    "seg_path": "/home/hamze/Documents/MedSAM/assets/",
    "box":[170, 37, 361, 163],
    "device":"cuda",
    "checkpoint":"/home/hamze/Documents/MedSAM/work_dir/MedSAM/medsam_vit_b.pth",

}
device = args["device"]
medsam_model = sam_model_registry["vit_b"](checkpoint=args["checkpoint"])
medsam_model = medsam_model.to(device)
medsam_model.eval()

config_path="configs/test_config.yaml"

if prompt_type ==1:
        text_prompt="lumbar multifidus. benign. malignant. pants." #1
elif prompt_type ==2:
    text_prompt="benign. malignant. pants. tumor." #2
elif prompt_type ==3:
    text_prompt='''a breast ultrasound scan with a benign lesion.
                    Find the malignant tumor in this breast ultrasound.'''
elif prompt_type ==4:
    text_prompt='''a breast ultrasound scan with a benign lesion.
                    Find the malignant tumor in this breast ultrasound.
                    A breast ultrasound scan showing a malignant tumor with irregular shape.
                    An ultrasound image showing a benign lesion with smooth contour.'''
elif prompt_type ==5:
    text_prompt='''a breast with a benign lesion.
                    Find the malignant tumor in this breast.
                    A breast showing a malignant tumor with irregular shape.
                    An image showing a benign lesion with smooth contour.'''
elif prompt_type ==6:
    text_prompt='''a breast with a benign lesion.
                    Find the malignant tumor in this breast.
                    A breast showing a malignant tumor with irregular shape.
                    An image showing a benign lesion with smooth contour.
                    benign cyst. 
                    malignant ductal carcinoma in situ.
                    malignant invasive ductal carcinoma.
                    malignant invasive lobular carcinoma.
                    malignant invasive lobular carcinoma with irregular shape.
                    malignant invasive lobular carcinoma with smooth contour.
                    malignant invasive lobular carcinoma with irregular shape.
                    malignant invasive lobular carcinoma with smooth contour.'''
    
# text_prompt="shirt .bag .pants"
data_config, model_config, training_config = ConfigurationManager.load_config(config_path)
model = load_model(model_config,training_config.use_lora)

text_prompt="lumbar multifidus." #1

caption = preprocess_caption(caption=text_prompt)

for img in os.listdir(data_config.val_dir):
    image_path=os.path.join(data_config.val_dir,img)
    #     image_source = Image.open(image_path).convert("RGB")
    image_source, image = load_image(image_path)
    h, w, _ = image_source.shape
    # pre = predict(model,
    #         image,
    #         caption,
    #         box_threshold,
    #         text_threshold,
    #         remove_combined= True)
    # print(img,pre)

    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        remove_combined=False
    )
    print(f"Original boxes size {boxes.shape}")
    if boxes.shape[0]>0:
        boxes, logits, phrases = apply_nms_per_phrase(image_source, boxes, logits, phrases)
        print(f"NMS boxes size {boxes.shape}")
    else:
        continue

    _, top2_indices = torch.topk(logits, top_k if boxes.shape[0]>=top_k else boxes.shape[0])

    boxes = boxes[top2_indices,:] * torch.Tensor([w, h, w, h])

    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    rec1 = xyxy

    phrases = [phrases[i] for i in top2_indices]
    
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

    # detections = sv.Detections(xyxy=xyxy)

    # labels = [
    #     f"{phrase} {logit:.2f}"
    #     for phrase, logit
    #     in zip(phrases, logits)
    # ]

    # box_annotator = sv.BoxAnnotator()
    # annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    # annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    # if len(boxes>0):
    #     rec = pre[0] * torch.tensor([w, h, w, h])
    #     rec = rec[0].cpu().numpy()
    #     rec1 = box_cxcywh_to_xyxy(rec)
    # else:
    #     rec1 = [0,0,w,h]

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
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Show the input image
    ax.imshow(img_3c)
    
    # Overlay all MedSAM masks on the same axis
    for color_index, medsam_seg in enumerate(medsam_segs):
        show_mask(medsam_seg, ax, color_index=color_index)
        show_box(np.array([int(x) for x in rec1_sorted[color_index]]), ax, 
                 label=f"{phrases_sorted[color_index]}",isBottom = color_index%2)

    ax.set_title(f"MedSAM {img}-{len(medsam_segs)}")
    ax.axis('off')
    plt.savefig(f"{save_path}/{img}", dpi=300, bbox_inches='tight')
    #plt.show()

#%%
