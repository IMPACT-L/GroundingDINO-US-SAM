#%%
from PIL import Image
import os
import cv2
import csv
import shutil
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
import sys
sys.path.append(os.path.abspath('..'))
from dataSaver import create_dataset
random.seed(42)
#%%
srcDir = '/home/hamze/Documents/Dataset/1-BreastDataset/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023'
dataset = 'breast'
#%%
rows = []
csvDir = f'{srcDir}/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.csv'
with open(csvDir, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    for row_ in reader:
        # print(row_[1],row_[20])
        if row_[2] != '':
            image_name = row_[1]
            mask_path = f"{srcDir}/BrEaST-Lesions_USG-images_and_masks/{image_name.replace('.png','_tumor.png')}"

            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                row=[
                    row_[20],
                    x, y, w, h,
                    f"{srcDir}/BrEaST-Lesions_USG-images_and_masks/{image_name}",
                    mask.shape[1],
                    mask.shape[0],
                    mask_path,
                    dataset
                ]
            
            rows.append(row)
        else :
            print('*'*20,row[1])

print(rows)
#%%
random.shuffle(rows)
total = len(rows)
train_size = int(0.7 * total)  # 70%
valid_size = int(0.2 * total)  # 20%
test_size = total - train_size - valid_size  # Remaining 10%

train_prompts = rows[:train_size]
valid_prompts = rows[train_size:train_size+valid_size]
test_prompts = rows[train_size+valid_size:]

#%%
create_dataset(train_prompts, 'train')
create_dataset(valid_prompts, 'val')
create_dataset(test_prompts, 'test')
# %%
