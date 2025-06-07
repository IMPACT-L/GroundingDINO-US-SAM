#%%
from PIL import Image
import os
import cv2
import csv
import shutil
import matplotlib.pyplot as plt
import random
import sys
sys.path.append(os.path.abspath('..'))
from dataSaver import create_dataset
random.seed(42)
#%%
srcDir = '/home/hamze/Documents/Dataset/1-BreastDataset/Breast_BUS_B_2024/BUS'
dataset = 'busb'
#%%
rows=[]
csvDir = f'{srcDir}/DatasetB.csv'
with open(csvDir, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    for row_ in reader:
        # print(row_[0],row_[1])
        image_name = f'{row_[0]}.png'
        mask_path = f"{srcDir}/GT/{image_name}"

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            row=[
                row_[1],
                x, y, w, h,
                f"{srcDir}/original/{image_name}",
                mask.shape[1],
                mask.shape[0],
                mask_path,
                dataset
            ]
        
        rows.append(row)

print(rows)
#%%
random.shuffle(rows)
print(rows)
total = len(rows)
train_size = int(0.7 * total)  # 70%
valid_size = int(0.2 * total)  # 20%
test_size = total - train_size - valid_size  # Remaining 10%

train_prompts = rows[:train_size]
valid_prompts = rows[train_size:train_size+valid_size]
test_prompts = rows[train_size+valid_size:]
create_dataset(train_prompts, 'train')
create_dataset(valid_prompts, 'val')
create_dataset(test_prompts, 'test')
#%%