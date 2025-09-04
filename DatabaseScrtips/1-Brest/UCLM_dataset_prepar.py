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
import sys
sys.path.append(os.path.abspath('..'))
from dataSaver import create_dataset
random.seed(42)
#%%
srcDir = '../Dataset/1-BreastDataset/BUS-UCLM'
dataset='busuclm'
#%%
mask_paths = glob.glob(f'{srcDir}/masks/*')
os.makedirs(f'{srcDir}/new_masks', exist_ok=True)
types = ['normal','benign','malignant']
benigins = []
malignants = []
for mask_path in mask_paths:
    mask = cv2.imread(mask_path) 
    file_name = mask_path.split('/')[-1]
    new_mask_path = f'{srcDir}/new_masks/{file_name}'
    
    for i in [1,2]:
        channel = mask[:, :, i]
        if channel.any():
            contours, _ = cv2.findContours(channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for j, contour in enumerate(contours[:2]):
                x, y, w, h = cv2.boundingRect(contour)
                row =[
                    types[i],
                    x, y, w, h,
                    mask_path.replace('masks','images'),
                    mask.shape[1],
                    mask.shape[0],
                    new_mask_path,
                    dataset,
                    j
                ]
                binary_mask = (channel > 0).astype(np.uint8)  # 0 and 1
                save_mask = (binary_mask * 255).astype(np.uint8)  # Scale to 0–255 for saving

                cv2.imwrite(new_mask_path, save_mask)
                malignants.append(row) if i==2 else benigins.append(row)
                # shutil.copy2(mask_path, new_mask_path)
            break
        # print(types[i])

# plt.imshow(mask)
#%%
random.shuffle(benigins)

total = len(benigins)
train_size = int(0.7 * total)  # 70%
valid_size = int(0.2 * total)  # 20%
test_size = total - train_size - valid_size  # Remaining 10%

# Split the numbers
train_list = benigins[:train_size]
valid_list = benigins[train_size:train_size+valid_size]
test_list = benigins[train_size+valid_size:]
print(len(train_list),len(valid_list),len(test_list))

create_dataset(train_list,'train')
create_dataset(valid_list,'val')
create_dataset(test_list,'test')

#%%
random.shuffle(malignants)

total = len(malignants)
train_size = int(0.7 * total)  # 70%
valid_size = int(0.2 * total)  # 20%
test_size = total - train_size - valid_size  # Remaining 10%

# Split the numbers
train_list = malignants[:train_size]
valid_list = malignants[train_size:train_size+valid_size]
test_list = malignants[train_size+valid_size:]
print(len(train_list),len(valid_list),len(test_list))
create_dataset(train_list,'train')
create_dataset(valid_list,'val')
create_dataset(test_list,'test')
# %%
