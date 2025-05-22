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

#%%
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

def prepare_batch(batch,device):
    images, targets = batch
    # # Convert list of images to NestedTensor and move to device
    if isinstance(images, (list, tuple)):
        images = nested_tensor_from_tensor_list(images)  # Convert list to NestedTensor
    images = images.to(device)

    captions=[]
    for target in targets:
        target['boxes']=target['boxes'].to(device)
        target['size']=target['size'].to(device)
        target['labels']=target['labels'].to(device)
        captions.append(target['caption'])
        
    return images, targets, captions
    

        

def process_images(
        model,
        text_prompt,
        data_config,
        box_threshold,
        text_threshold,
        save_path=''
):
    # visualizer = GroundingDINOVisualizer(save_dir="visualizations")

    for img in os.listdir(data_config.val_dir):
        image_path=os.path.join(data_config.val_dir,img)
        image_source, image = load_image(image_path)
        # visualizer.visualize_image(model,image,text_prompt,image_source,img,box_th=box_threshold,txt_th=text_threshold)

        boxes, logits, phrases = predict(
           model=model,
           image=image,
           caption=text_prompt,
           box_threshold=box_threshold,
           text_threshold=text_threshold,
           remove_combined=True
        )
        print(f"Original boxes size {boxes.shape}")
        if boxes.shape[0]>0:
           boxes, logits, phrases = apply_nms_per_phrase(image_source, boxes, logits, phrases)
           print(f"NMS boxes size {boxes.shape}")
        
        top_k=10
        _, top2_indices = torch.topk(logits, top_k if boxes.shape[0]>=top_k else boxes.shape[0])

        annotated_frame = annotate(image_source=image_source, 
                                   boxes=boxes[top2_indices,:], 
                                   logits=logits[top2_indices], 
                                   phrases=[phrases[i] for i in top2_indices]
                                   )
        
        cv2.imwrite(f"{save_path}/{img}", annotated_frame)

#%%
if __name__ == "__main__":
    box_threshold=0.1
    text_threshold=0.1
    prompt_type = 4
    save_path = f"visualizations/Sentences/BB/bb_{prompt_type}_{box_threshold}_{text_threshold}"
    # shutil.rmtree ("visualizations/inference")
    os.makedirs(save_path, exist_ok=True)

    # Config file of the prediction, the model weights can be complete model weights but if use_lora is true then lora_wights should also be present see example
    ## config file
    config_path="configs/test_config.yaml"
    
    
    if prompt_type ==1:
        text_prompt="benign. malignant. pants." #1
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
    

    data_config, model_config, test_config = ConfigurationManager.load_config(config_path)
    model = load_model(model_config,test_config.use_lora)

    test_dataset = GroundingDINODataset(
        data_config.val_dir,
        data_config.val_ann
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,  # Keep batch size 1 for validation
        shuffle=False,
        num_workers=1,
        collate_fn=lambda x: tuple(zip(*x))
    )
    device = 'cuda'
    model.to(device)
    model.eval()
    val_losses = defaultdict(float)

    class_loss_coef=1.0
    bbox_loss_coef=5.0  
    giou_loss_coef=1.0  
    eos_coef=0.1
    max_txt_len=256

    matcher=build_matcher(set_cost_class=class_loss_coef*2,
            set_cost_bbox=bbox_loss_coef,
            set_cost_giou=giou_loss_coef)
            
    losses = ['labels', 'boxes']
    weights_dict= {'loss_ce': class_loss_coef, 'loss_bbox': bbox_loss_coef, 'loss_giou': giou_loss_coef}
    # Give more weightage to bobx loss in loss calculation compared to matcher 
    weights_dict_loss = {'loss_ce': class_loss_coef, 'loss_bbox': bbox_loss_coef*2, 'loss_giou': giou_loss_coef}
    criterion = SetCriterion(max_txt_len, matcher, eos_coef, losses)
    criterion.to(device)
    results = []
    num_batches = 0
    for batch in test_loader:
        images, targets, captions = prepare_batch(batch,device)
        outputs = model(images, captions=captions)
        
        # Calculate losses
        loss_dict = criterion(outputs, targets, captions=captions, tokenizer=model.tokenizer)
        
        # Accumulate losses
        txt=""
        for k, v in loss_dict.items():
            val_losses[k] += v.item()
            txt+=f'({k},{v:.4f}),'
        
        val_losses['total_loss'] += sum(loss_dict[k] * weights_dict[k] 
                                    for k in loss_dict.keys() if k in weights_dict_loss).item()
        print(txt)  
        results.append(txt)
        num_batches+=1

    results.append('_'*50)

    # Average losses
    last_row = {k: f'{v/num_batches:.4f}' for k, v in val_losses.items()}
    print(last_row)
    results.append(last_row)
    
    
    with open(f"{save_path}/results.txt", "w") as f:  # Use "a" to append instead of overwrite
        for line in results:
            f.write(str(line) + "\n")

    process_images(model,text_prompt,data_config,
                   box_threshold=box_threshold,
                   text_threshold=text_threshold,
                   save_path=save_path)
#%%