#%%
# python -m DatabaseScrtips.5-Liver.AUL_dataset_prepar

from PIL import Image
import os
import cv2
import csv
import shutil
import matplotlib.pyplot as plt
import random
import glob
import numpy as np
import json
import matplotlib.patches as patches
import sys
sys.path.append(os.path.abspath('..'))
from dataSaver import create_dataset
random.seed(42)
#%%
srcDir = '/home/hamze/Documents/Dataset/5-Liver/AUL'
dataset = 'aul'
#%%
def getData(in_type,show_plot=False):
    rows = []
    image_paths = glob.glob(f'{srcDir}/{in_type}/image/*')

    for image_path in image_paths:
        img_name = image_path.split('/')[-1]
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mass_path = f"{srcDir}/{in_type}/segmentation/mass/{img_name.replace('jpg','json')}"

        with open(mass_path, 'r') as f:
            mass_data = json.load(f)

        mass_points = np.array(mass_data)
        mass_polygon = patches.Polygon(mass_points, closed=True, edgecolor='green', facecolor='none', linewidth=2)

        x_min = int(np.min(mass_points[:, 0]))
        x_max = int(np.max(mass_points[:, 0]))
        y_min = int(np.min(mass_points[:, 1]))
        y_max = int(np.max(mass_points[:, 1]))

        mask = np.zeros(img.shape[:2], dtype=np.uint8)  # single channel

        mass_points_int = np.array([mass_points], dtype=np.int32)

        cv2.fillPoly(mask, mass_points_int, color=255)
        if show_plot:
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            plt.title(img_name)
            plt.gca().add_patch(mass_polygon)

            bbox = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
            plt.gca().add_patch(bbox)

            # Visualize the mask
            plt.imshow(mask, cmap='gray',alpha=.5)
            plt.title("Mass Mask")
            plt.axis('off')
            plt.show()

        # Optionally save the mask
        mask_save_path = os.path.join(srcDir, "masks", f"{dataset}_{img_name.replace('.jpg', '.png')}")
        cv2.imwrite(mask_save_path, mask)
        row =[
            in_type.lower(),
            x_min, y_min, x_max-x_min, y_max - y_min,
            image_path,
            mask.shape[1],
            mask.shape[0],
            mask_save_path,
            dataset
        ]
        rows.append(row)
    return rows
#%%
malignants = getData('Malignant')
random.shuffle(malignants)

total = len(malignants)
train_size = int(0.7 * total)  # 70%
valid_size = int(0.2 * total)  # 20%
test_size = total - train_size - valid_size  # Remaining 10%

train_numbers = malignants[:train_size]
valid_numbers = malignants[train_size:train_size+valid_size]
test_numbers = malignants[train_size+valid_size:]
print(len(train_numbers),len(valid_numbers),len(test_numbers))

create_dataset(train_numbers,'train')
create_dataset(valid_numbers,'val')
create_dataset(test_numbers,'test')
# cv2.imwrite(mask_save_path, mask)
# %%
benigns =  getData('Benign')
random.shuffle(benigns)

total = len(benigns)
train_size = int(0.7 * total)  # 70%
valid_size = int(0.2 * total)  # 20%
test_size = total - train_size - valid_size  # Remaining 10%

train_numbers = benigns[:train_size]
valid_numbers = benigns[train_size:train_size+valid_size]
test_numbers = benigns[train_size+valid_size:]
print(len(train_numbers),len(valid_numbers),len(test_numbers))
create_dataset(train_numbers,'train')
create_dataset(valid_numbers,'val')
create_dataset(test_numbers,'test')
#%%