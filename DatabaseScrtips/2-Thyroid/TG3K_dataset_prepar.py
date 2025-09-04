#%%
from PIL import Image
import os
import cv2
import csv
import shutil
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
import glob
import numpy as np
import sys
sys.path.append(os.path.abspath('..'))
from dataSaver import create_dataset
random.seed(42)
#%%
srcDir = '../Dataset/2-Thyroid-Dataset/tg3k'
dataset = 'tg3k'
#%%
# folder_path = '../multimodal-data/USDATASET/images/val'
# for filename in os.listdir(folder_path):
#     if 'tn3k' in filename:
#         new_name = filename.replace('tn3k', 'tg3k')
#         src = os.path.join(folder_path, filename)
#         dst = os.path.join(folder_path, new_name)
#         os.rename(src, dst)
#         print(f'Renamed: {filename} -> {new_name}')
#%%
mask_paths = glob.glob(f'{srcDir}/thyroid-mask/*')
min_area = 100
min_area=500
data = []
for mask_path in mask_paths:
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask[mask>0]=255

    file_name = mask_path.split('/')[-1]

    img_path = f"{mask_path.replace('thyroid-mask','thyroid-image')}"
    img = cv2.imread(f'{img_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.title(file_name)
    # plt.imshow(mask)
    # plt.show()
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [r for r in contours if cv2.contourArea(r) >= min_area]

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for i, contour in enumerate(contours[:2]):
        x, y, w, h = cv2.boundingRect(contour)
        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(img)
        # ax[1].imshow(mask)
        # rect = patches.Rectangle((x, y), w, h,
        #                         linewidth=2, edgecolor='r', facecolor='none')
        # ax[1].add_patch(rect)
        # plt.title("Bounding Box")
        # # plt.axis('off')
        # plt.show()
        row =[
            'thyroid tumor',
            x, y, w, h,
            img_path,
            mask.shape[1],
            mask.shape[0],
            mask_path,
            dataset,
            i
        ]
        # print(row)
        data.append(row)

    # print(contour.shape)
    # break

print(data[0])

#%%
random.shuffle(data)
print(data)
total = len(data)
train_size = int(0.7 * total)  # 70%
valid_size = int(0.2 * total)  # 20%
test_size = total - train_size - valid_size  # Remaining 10%

train_data = data[:train_size]
valid_data = data[train_size:train_size+valid_size]
test_data = data[train_size+valid_size:]

#%%
create_dataset(train_data, 'train')
create_dataset(valid_data, 'val')
create_dataset(test_data, 'test')
# %%
