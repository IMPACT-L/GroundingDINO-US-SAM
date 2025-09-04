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
import sys
sys.path.append(os.path.abspath('..'))
from dataSaver import create_dataset
random.seed(42)
#%%
srcDir = '../Dataset/1-BreastDataset/STU-Hospital/'
dataset = 'stu'
#%%
mask_paths = glob.glob(f'{srcDir}/mask_*')
data = []
for mask_path in mask_paths:
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    file_name = mask_path.split('/')[-1]

    img_path = f"{mask_path.replace('mask','Test_Image')}"
    # img = cv2.imread(f'{img_path}')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for _, contour in enumerate(contours):
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
            'tumor',
            x, y, w, h,
            img_path,
            mask.shape[1],
            mask.shape[0],
            mask_path,
            dataset
        ]
       
        data.append(row)

print(data)

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
create_dataset(train_data, 'train')
create_dataset(valid_data, 'val')
create_dataset(test_data, 'test')
# %%
