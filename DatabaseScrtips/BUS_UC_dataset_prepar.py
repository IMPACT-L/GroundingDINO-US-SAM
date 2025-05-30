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
random.seed(42)
#%%
srcDir = '/home/hamze/Documents/Dataset/BUS_UC'
desDir = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/USDATASET'
#%%
os.makedirs(f'{desDir}/images/train', exist_ok=True)
os.makedirs(f'{desDir}/images/val', exist_ok=True)
os.makedirs(f'{desDir}/images/test', exist_ok=True)
#%%
paths = glob.glob(f'{srcDir}/Malignant/masks/*')
for path in paths:
    os.rename(path, path.replace('_bus_us','_malignant_bus_uc'))
#%%
paths = glob.glob(f'{srcDir}/Malignant/images/*')
for path in paths:
    os.rename(path, path.replace('_bus_us','_malignant_bus_uc'))
#%%
paths = glob.glob(f'{srcDir}/Benign/masks/*')
for path in paths:
    os.rename(path, path.replace('_bus_us','_benign_bus_uc'))
#%%
paths = glob.glob(f'{srcDir}/Benign/images/*')
for path in paths:
    os.rename(path, path.replace('_bus_us','_benign_bus_uc'))
#%%
mask_paths = glob.glob(f'{srcDir}/Malignant/masks/*')
malignants = []
for mask_path in mask_paths:
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    file_name = mask_path.split('/')[-1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for _, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        row =[
            'malignant',
            x, y, w, h,
            file_name,
            mask.shape[1],
            mask.shape[0],
            mask_path,
            'BUS_UC'
        ]
       
        malignants.append(row)
        # shutil.copy2(mask_path, new_mask_path)
    # break
print(malignants)
# os.rename(benigin_mask_path, benigin_mask_path.replace('.png','_bus_us.png'))
#%%
mask_paths = glob.glob(f'{srcDir}/Benign/masks/*')
benigins = []
for mask_path in mask_paths:
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    file_name = mask_path.split('/')[-1]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for _, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        row =[
            'benigin',
            x, y, w, h,
            file_name,
            mask.shape[1],
            mask.shape[0],
            mask_path,
            'BUS_UC'
        ]
       
        benigins.append(row)
        # shutil.copy2(mask_path, new_mask_path)
    # break
print(benigins)
#%%
def create_dataset(type, number_list, output_type, firstRow=False):
    with open( f'{desDir}/{output_type}_annotation.CSV', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if firstRow:
            writer.writerow(['label_name', 'bbox_x', 'bbox_y', 
                        'bbox_width', 'bbox_height', 
                        'image_name', 'image_width', 'image_height','mask_path','dataset'])
        
        for row in number_list:
            try:
                writer.writerow(row)
                image_name = row[5]
                img_path = f'{srcDir}/{type}/images/{image_name}'
                dst = f'{desDir}/images/{output_type}/{image_name}'
                
                shutil.copy2(img_path, dst)
                print('saved')
               
            except Exception as e:
                print(f"Error processing {row}: {str(e)}")

#%%
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

create_dataset('Benign', train_numbers,'train',False)
create_dataset('Benign', valid_numbers,'val',False)
create_dataset('Benign', test_numbers,'test',False)

#%%
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
create_dataset('Malignant', train_numbers,'train',False)
create_dataset('Malignant', valid_numbers,'val',False)
create_dataset('Malignant', test_numbers,'test',False)
# %%
