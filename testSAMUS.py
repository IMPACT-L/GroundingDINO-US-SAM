#%%
import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import cv2
from PIL import Image
from sklearn.metrics import jaccard_score, f1_score
import warnings
warnings.filterwarnings("ignore")
#%%
def sklearn_iou(pred_mask, true_mask):
    return jaccard_score(true_mask.flatten(), pred_mask.flatten())

def sklearn_dice(pred_mask, true_mask):
    return f1_score(true_mask.flatten(), pred_mask.flatten())

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
                    }
    return textCSV
#%%
test_path = f'multimodal-data/test_image'
csvPath = 'multimodal-data/test.CSV'

datasets = ["breast", "buid", "busuc","busuclm","busb", "busi",
            "stu","s1","tn3k","tg3k","105us",
            "aul","muregpro","regpro","kidnyus"]

for selectedDataset in datasets:
    print("*"*20,selectedDataset,"*"*20)

    save_result_path = f'visualizations/SAMUS/{selectedDataset}'
    os.makedirs(save_result_path, exist_ok=True)
    textCSV = getTextSample(selectedDataset)

    show_plots = True
    ious = []
    dices = []
    ious_after = []
    threshold = .5
    for image_index,image_name in enumerate(textCSV):
        # samus_path = f'visualizations/GroundedSAM-US_unseen/SAMUS/{selectedDataset}/{image_name}'.replace('png','npz').replace('jpg','npz').replace('.bmp','.npz').replace('.tif','.npz')

        sam_path = f'multimodal-data/SAMUS/{selectedDataset}/{image_name}'.replace('png','npz').replace('jpg','npz').replace('.bmp','.npz').replace('.tif','.npz')
        # if not os.path.exists(sam_path):
        #     continue
        image_path=os.path.join(test_path,image_name)
        image_source = Image.open(image_path).convert('RGB')
        image_source = np.asarray(image_source)
        mask_path = os.path.join(test_path.replace('test_image','test_mask'),image_name)
        mask_source = Image.open(mask_path).convert('L')
        mask_source = np.asarray(mask_source).copy()
    
        mask_source[mask_source>=threshold]=1
        mask_source[mask_source<threshold]=0

        # sam_path = f'multimodal-data/MedClipSamResults/MedCLIP-SAMv2/{selectedDataset}/{image_name}'.replace('png','npz').replace('jpg','npz').replace('.bmp','.npz')
        data = np.load(sam_path)
        sam_mask = data['prediction']  # Replace 'array_name' with actual key from data.files
        # if sam_mask.shape[0]!=image_source.shape[0] or sam_mask.shape[1]!=image_source.shape[1]:
        #     sam_mask = cv2.resize(sam_mask.astype(np.uint8), (image_source.shape[1], image_source.shape[0]), interpolation=cv2.INTER_NEAREST)

        if image_source.shape[0]!=sam_mask.shape[0] or image_source.shape[1]!=sam_mask.shape[1]:
            image_source = cv2.resize(image_source.astype(np.uint8), (sam_mask.shape[1], sam_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        if mask_source.shape[0]!=sam_mask.shape[0] or mask_source.shape[1]!=sam_mask.shape[1]:
            mask_source = cv2.resize(mask_source.astype(np.uint8), (sam_mask.shape[1], sam_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        sam_mask[sam_mask>=threshold]=1
        sam_mask[sam_mask<threshold]=0


        iou = sklearn_iou(sam_mask,mask_source)*100
        dic = sklearn_dice(sam_mask,mask_source)*100
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
            tmp_image[:,:,2][sam_mask==1]=255
            ax[2].set_title(f'iou: {iou:.2f}, dice: {dic:.2f}')
            ax[2].axis('off')
            ax[2].imshow(tmp_image)
            plt.savefig(f"{save_result_path}/{image_name.replace('.bmp','.png')}") 
            plt.close()
            # plt.show(block=False)

            ious.append(iou)
            dices.append(dic)

    ious = np.array(ious)
    dices = np.array(dices)
    print(f"Average IoU: {ious.mean():.2f}±{ious.std():.2f}")
    print(f"Average Dic: {dices.mean():.2f}±{dices.std():.2f}")
    print(f"Min IoU[{1+ious.argmin()}]: {ious.min():.2f}")
    print(f"Max IoU[{1+ious.argmax()}]: {ious.max():.2f}")
    with open(f'{save_result_path}/result.txt', 'w') as f:
        f.write(f"Average Dice, IoU: {dices.mean():.2f}±{dices.std():.0f} & {ious.mean():.2f}±{ious.std():.0f}\n")
print('Finished')
    # %%
