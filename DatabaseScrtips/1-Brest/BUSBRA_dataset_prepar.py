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
srcDir = '/home/hamze/Documents/Dataset/1-BreastDataset/BUSBRA'
dataset = 'busbra'

#%%
rows=[]
csvDir = f'{srcDir}/bus_data.csv'
with open(csvDir, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header row
    for row_ in reader:
        print(row_[0],row_[3],row_[6],row_[7],row_[9])
        
        image_name = f'{row_[0]}.png'

        mask_path = f"{srcDir}/Masks/{image_name.replace('bus','mask')}"

        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        bbox = list(map(int, row_[9].strip('[]').split(',')))
        x, y, w, h = bbox
        
        row=[
            row_[3],
            x, y, w, h,
            f"{srcDir}/Images/{image_name}",
            row_[3],
            row_[4],
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