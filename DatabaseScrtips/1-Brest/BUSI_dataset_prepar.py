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
srcDir = '/home/hamze/Documents/Dataset/1-BreastDataset/BreastBUSI_Images'
dataset = 'busi'
#%%
def getData(data_type):
    mask_paths = glob.glob(f'{srcDir}/{data_type}/*_mask.png')
    data_list = []
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for _, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            row =[
                data_type.lower(),
                x, y, w, h,
                mask_path.replace('_mask',''),
                mask.shape[1],
                mask.shape[0],
                mask_path,
                dataset
            ]
        
            data_list.append(row)
    return data_list
#%%
malignants = getData('malignant')
random.shuffle(malignants)

total = len(malignants)
train_size = int(0.7 * total)  # 70%
valid_size = int(0.2 * total)  # 20%
test_size = total - train_size - valid_size  # Remaining 10%

# Split the numbers
train_numbers = malignants[:train_size]
valid_numbers = malignants[train_size:train_size+valid_size]
test_numbers = malignants[train_size+valid_size:]
print(len(train_numbers),len(valid_numbers),len(test_numbers))
create_dataset(train_numbers,'train')
create_dataset(valid_numbers,'val')
create_dataset(test_numbers,'test')

#%%
benigins = getData('benign')

random.shuffle(benigins)

total = len(benigins)
train_size = int(0.7 * total)  # 70%
valid_size = int(0.2 * total)  # 20%
test_size = total - train_size - valid_size  # Remaining 10%

# Split the numbers
train_numbers = benigins[:train_size]
valid_numbers = benigins[train_size:train_size+valid_size]
test_numbers = benigins[train_size+valid_size:]
print(len(train_numbers),len(valid_numbers),len(test_numbers))

create_dataset(train_numbers,'train')
create_dataset(valid_numbers,'val')
create_dataset(test_numbers,'test')
# %%
