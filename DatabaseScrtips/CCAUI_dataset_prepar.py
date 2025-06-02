#%%
from PIL import Image
import os
import cv2
import csv
import shutil
import matplotlib.pyplot as plt
import random
import glob
random.seed(42)
#%%
srcDir = '/home/hamze/Documents/Dataset/CCAUI'
desDir = '../multimodal-data/USDATASET'
#%%
os.makedirs(f'{desDir}/images/train', exist_ok=True)
os.makedirs(f'{desDir}/images/val', exist_ok=True)
os.makedirs(f'{desDir}/images/test', exist_ok=True)
dataset = 'ccaui'
#%%
mask_paths = glob.glob(f'{srcDir}/Mask/*')
data = []
for mask_path in mask_paths:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image_name = mask_path.split('/')[-1]
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        print(f'Contour {i}: x={x}, y={y}, w={w}, h={h}')
        contour = contour.squeeze()  # shape becomes (N, 2)

        contour = contour.reshape(-1, 2)
        # plt.imshow(mask, cmap='gray')
        # plt.plot(contour[:, 0], contour[:, 1], color='lime')  # plot x vs y
        # plt.scatter([x, x+w, x+w, x], [y, y, y+h, y+h], color=['red', 'green', 'blue', 'yellow'])
        # plt.show()

        data.append([
                        'carotid artery',
                        x, y, w, h,
                        f'{dataset}_{image_name}',
                        mask.shape[1],
                        mask.shape[0],
                        mask_path,
                        dataset
                    ]
                )   
    # break

#%%
total = len(data)
train_size = int(0.7 * total)  # 70%
valid_size = int(0.2 * total)  # 20%
test_size = total - train_size - valid_size  # Remaining 10%

train_prompts = data[:train_size]
valid_prompts = data[train_size:train_size+valid_size]
test_prompts = data[train_size+valid_size:]
#%%
def create_dataset(prompts_in, output_type):
    with open( f'{desDir}/{output_type}_annotation.CSV', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(prompts_in)
##
create_dataset(train_prompts, 'train')
create_dataset(valid_prompts, 'val')
create_dataset(test_prompts, 'test')
# %%
def copyImages(prompts_in, output_type):
    for prompt in prompts_in:
        image_name = prompt[5].replace(dataset+'_','')
        img_path = f'{srcDir}/Image/{image_name}'
        dst = f'{desDir}/images/{output_type}/{dataset}_{image_name}'
        shutil.copy2(img_path, dst)

# %%
copyImages(train_prompts, 'train')
copyImages(valid_prompts, 'val')
copyImages(test_prompts, 'test')
# %%
