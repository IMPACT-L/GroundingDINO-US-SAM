#%%
from PIL import Image
import os
import cv2
import csv
import shutil
import matplotlib.pyplot as plt
import glob
import random
import sys
sys.path.append(os.path.abspath('..'))
from dataSaver import create_dataset
random.seed(42)
#%%
srcDir = '/home/hamze/Documents/Dataset/LUMINOUS_Database'
dataset = 'luminous'
#%%
rows=[]
# img_path = f'{srcDir}/B-mode/{image_name}'
# mask_path = f"{srcDir}/Masks/{image_name.replace('Bmode','Mask')}"
           
mask_paths = glob.glob(f'{srcDir}/masks/*')

for mask_path in mask_paths:
    mask = cv2.imread(mask_path) 
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    file_name = mask_path.split('/')[-1]
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour in enumerate(contours[:2]):
        x, y, w, h = cv2.boundingRect(contour)

        print(f'Contour {i}: x={x}, y={y}, w={w}, h={h}')
        contour = contour.squeeze()  # shape becomes (N, 2)

        contour = contour.reshape(-1, 2)

        plt.imshow(mask, cmap='gray')
        plt.plot(contour[:, 0], contour[:, 1], color='lime')  # plot x vs y
        plt.scatter([x, x+w, x+w, x], [y, y, y+h, y+h], color=['red', 'green', 'blue', 'yellow'])
        plt.title(f"{file_name},{'left' if x+w/2 < mask.shape[1]/2 else 'right'}")
        plt.show()
        
        row =[
            f"{'left' if x+w/2 < mask.shape[1]/2 else 'right'} {'lower back muscle'}",
            x, y, w, h,
            mask_path.replace('masks','images').replace('Mask','Bmode'),
            mask.shape[1],
            mask.shape[0],
            mask_path,
            dataset
        ]
        rows.append(row)
    # break
#%%
# random.shuffle(prompts)
# print(prompts)
# total = len(prompts)
# train_size = int(0.7 * total)  # 70%
# valid_size = int(0.2 * total)  # 20%
# test_size = total - train_size - valid_size  # Remaining 10%

# train_prompts = prompts[:train_size]
# valid_prompts = prompts[train_size:train_size+valid_size]
# test_prompts = prompts[train_size+valid_size:]
##
# create_dataset(train_prompts, 'train')
# create_dataset(valid_prompts, 'val')
create_dataset(rows, 'test')

# %%
