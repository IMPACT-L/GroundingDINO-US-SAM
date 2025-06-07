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
srcDir = '/home/hamze/Documents/Dataset/3-Kidny/kidneyUS'
dataset = 'kidnyus'
#%%
# Capsule = 0, Central Echo Complex = 1, Medulla = 2, Cortex = 3
annotations = ['capsule', 'central echo complex' ,'medulla', 'cortex']
csvDir = f'{srcDir}/kidneyUS-main/labels/reviewed_labels_2.csv'
rows = []
with open(csvDir, 'r', newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) 
    for row_ in reader:
        index = int(row_[4])
        if index<4 and row_[6] != '{}':
            print(row_[0],row_[3],row_[4],annotations[index])
            image_path=f'{srcDir}/images/{row_[0]}'
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            regions = 'regions'
            if index == 0:
                regions = 'capsule'

            mask_path = f'{srcDir}/kidneyUS-main/labels/reviewed_masks_2/{regions}/{row_[0]}'
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            
            # if index != 0:
            #     mask[mask!=index] = 0
            print(f'{row_[0]},{annotations[index]},{np.unique(mask)}')
            print('*'*30)
            img_new_path = f"{srcDir}/kidneyUS-main/new_image/{row_[0].replace('.png','')}_{annotations[index]}.png"
            mask_new_path = f"{srcDir}/kidneyUS-main/new_mask/{row_[0].replace('.png','')}_{annotations[index]}.png"
            shutil.copy2(image_path, img_new_path)
            cv2.imwrite(mask_new_path,mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                row=[
                    f'kidney {annotations[index]}',
                    x, y, w, h,
                    img_new_path,
                    mask.shape[1],
                    mask.shape[0],
                    mask_new_path,
                    dataset
                ]
                rows.append(row)
            # shutil.copy2(image_path, f'{srcDir}/kidneyUS-main/new_image/{row_[0]}_{index}')

            # plt.title(f'{row_[0]},{annotations[index]},{np.unique(mask)}')
            # plt.imshow(img)
            # # plt.imshow(capsule,alpha=.2)
            # plt.imshow(mask,alpha=.2)
            # plt.show()


        # for annotation in annotations:
        #     if annotation in row_[6].lower():
        #         annotation = 'kidny '+annotation
        #         print(row_[0],row_[3],row_[4],annotation)
                

        # break
#%%
# image_paths = glob.glob(f'{srcDir}/images/*')

# for image_path in image_paths[9:10]:
#     file_name=image_path.split('/')[-1]
#     img = cv2.imread(image_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     plt.imshow(img)
#     mask_path = f'{srcDir}/kidneyUS-main/labels/reviewed_masks_2/regions/{file_name}'
#     mask = cv2.imread(mask_path)
#     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

#     capsule_path = f'{srcDir}/kidneyUS-main/labels/reviewed_masks_2/capsule/{file_name}'
#     capsule = cv2.imread(capsule_path)
#     capsule = cv2.cvtColor(capsule, cv2.COLOR_BGR2GRAY)
#     # mask[mask!=1]=0
#     plt.title(f"{file_name},{np.unique(capsule)},{np.unique(mask)}")
#     # plt.imshow(capsule,alpha=.2)
#     plt.imshow(mask,alpha=.2)
#     plt.show()
    
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
