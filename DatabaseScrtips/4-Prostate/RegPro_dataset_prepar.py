#%%
from PIL import Image
import os
import cv2
import csv
import shutil
import matplotlib.pyplot as plt
import random
import glob
import numpy as np
import nibabel as nib
import sys
sys.path.append(os.path.abspath('..'))
from dataSaver import create_dataset
random.seed(42)
#%%
srcDir = '../Dataset/4-Prostate/regPro'
dataset = 'regpro'
#%%

def getData(in_type):
# in_type = 'val' # 'train'
    img_paths = glob.glob(f'{srcDir}/{in_type}/us_images/*')
    os.makedirs(f'{srcDir}/saved_images', exist_ok=True)
    os.makedirs(f'{srcDir}/saved_masks', exist_ok=True)

    # Changed to list (rows was using .add() which is for sets)
    rows = []

    for i, img_path in enumerate(img_paths):
        mask_path = img_path.replace('us_images', 'us_labels')
        
        # Load image and mask
        mask = nib.load(mask_path).get_fdata()
        img = nib.load(img_path).get_fdata()
        # print("Data shape:", img.shape, mask.shape)

        for slice_idx in range(mask.shape[2]):
            if np.any(mask[:, :, slice_idx,0]):  # Added np. for clarity
                img_slice_norm = cv2.normalize(img[:, :, slice_idx], None, 0, 255, cv2.NORM_MINMAX)
                img_slice_uint8 = img_slice_norm.astype(np.uint8)
                mask_slice_uint8 = mask[:, :, slice_idx,0].astype(np.uint8)
                
                # plt.imshow(img[:,:,slice_idx,0],cmap='gray')
                # plt.imshow(mask[:,:,slice_idx,0],alpha=.2)
                # plt.show()
                
                contours, _ = cv2.findContours(mask_slice_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for k,contour in enumerate(contours):  # Removed unused index _
                    x, y, w, h = cv2.boundingRect(contour)
                    # print(k)
                    
                    msk = np.zeros_like(mask_slice_uint8)
                    msk[y:y+h,x:x+w] = mask_slice_uint8[y:y+h,x:x+w]

                    
                    img_filename = f'{srcDir}/saved_images/{i}_{k}_{in_type[0]}_{slice_idx}.png'
                    mask_filename = f'{srcDir}/saved_masks/{i}_{k}_{in_type[0]}_{slice_idx}.png'
                
                    cv2.imwrite(mask_filename, msk)
                    cv2.imwrite(img_filename, img_slice_uint8)
                    # Append to list instead of add
                    rows.append([  # Changed from .add() to .append()
                        'prostate',
                        x, y, w, h,
                        img_filename,
                        mask.shape[1],  # width
                        mask.shape[0],  # height
                        mask_filename,
                        dataset  # Make sure 'dataset' variable is defined
                    ])
        # break
    return rows
#%%
rows = getData('val')
rows += getData('train')

random.shuffle(rows)

total = len(rows)
train_size = int(0.7 * total)  # 70%
valid_size = int(0.2 * total)  # 20%
test_size = total - train_size - valid_size  # Remaining 10%

# Split the numbers
train_numbers = rows[:train_size]
valid_numbers = rows[train_size:train_size+valid_size]
test_numbers = rows[train_size+valid_size:]
print(len(train_numbers),len(valid_numbers),len(test_numbers))
create_dataset(train_numbers,'train')
create_dataset(valid_numbers,'val')
create_dataset(test_numbers,'test')
# %%
