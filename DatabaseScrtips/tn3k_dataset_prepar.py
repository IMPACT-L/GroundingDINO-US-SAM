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
random.seed(42)
#%%
srcDir = '/home/hamze/Documents/Dataset/2-Thyroid Dataset/tn3k'
desDir = '../multimodal-data/USDATASET'
dataset = 'tn3k'
#%%
os.makedirs(f'{desDir}/images/train', exist_ok=True)
os.makedirs(f'{desDir}/images/val', exist_ok=True)
os.makedirs(f'{desDir}/images/test', exist_ok=True)
#%%
mask_paths = glob.glob(f'{srcDir}/thyroid-mask/*')
data = []
for mask_path in mask_paths:
    mask = cv2.imread(mask_path)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    file_name = mask_path.split('/')[-1]

    img_path = f"{mask_path.replace('thyroid-mask','thyroid-image')}"
    img = cv2.imread(f'{img_path}')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.title(file_name)
    # plt.imshow(mask)
    # plt.show()
    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # for _, contour in enumerate(contours):
    # x, y, w, h = cv2.boundingRect(contour)

    fig, ax = plt.subplots(1,2)

    ax[0].imshow(img)

    ax[1].imshow(mask)
    # rect = patches.Rectangle((x, y), w, h,
                            # linewidth=2, edgecolor='r', facecolor='none')
    # ax[1].add_patch(rect)

    plt.title("Bounding Box")
    # plt.axis('off')
    plt.show()
    break
    #     row =[
    #         'tumor',
    #         x, y, w, h,
    #         f"{dataset}_{file_name.replace('thyroid-mask','thyroid-image')}",
    #         mask.shape[1],
    #         mask.shape[0],
    #         mask_path,
    #         dataset
    #     ]
       
    #     data.append(row)

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
#%%
def create_dataset(dat_in, output_type):
    with open( f'{desDir}/{output_type}_annotation.CSV', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(dat_in)  


#%%
create_dataset(train_data, 'train')
create_dataset(valid_data, 'val')
create_dataset(test_data, 'test')
# %%
def copyImages(data_in, output_type):
    for data in data_in:
        image_name = data[5].replace(dataset+'_','')

        img_path = f"{srcDir}/{image_name.replace('test_image','Test_Image')}"
        dst = f'{desDir}/images/{output_type}/{dataset}_{image_name}'
        shutil.copy2(img_path, dst)
        print(dst)


# %%
copyImages(train_data, 'train')
copyImages(valid_data, 'val')
copyImages(test_data, 'test')
# %%
