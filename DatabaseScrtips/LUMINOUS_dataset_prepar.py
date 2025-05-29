#%%
from PIL import Image
import os
import cv2
import csv
import shutil
import matplotlib.pyplot as plt
import random
random.seed(42)
#%%
srcDir = '/home/hamze/Documents/Dataset/LUMINOUS_Database'
desDir = '../multimodal-data/Breast'
#%%
os.makedirs(f'{desDir}/images/train', exist_ok=True)
os.makedirs(f'{desDir}/images/val', exist_ok=True)
os.makedirs(f'{desDir}/images/test', exist_ok=True)
#%%
def readTextPrompt(prompt_dir):
    prompts = []
    with open(prompt_dir, 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for columns in reader:
            prompts.append((columns[0],columns[1]))
    return prompts
prompts = readTextPrompt('luminous_prompts.csv')
print(prompts)
random.shuffle(prompts)
print(prompts)
total = len(prompts)
train_size = int(0.7 * total)  # 70%
valid_size = int(0.2 * total)  # 20%
test_size = total - train_size - valid_size  # Remaining 10%

train_prompts = prompts[:train_size]
valid_prompts = prompts[train_size:train_size+valid_size]
test_prompts = prompts[train_size+valid_size:]
#%%
def create_dataset(prompts_in, output_type, firstRow=False):
    with open( f'{desDir}/{output_type}_annotationtest.CSV', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if firstRow:
            writer.writerow(['label_name', 'bbox_x', 'bbox_y', 
                    'bbox_width', 'bbox_height', 
                    'image_name', 'image_width', 'image_height','mask_path'])
                
        for prompt in prompts_in:
            image_name = prompt[0]
            img_path = f'{srcDir}/B-mode/{image_name}'
            mask_path = f"{srcDir}/Masks/{image_name.replace('Bmode','Mask')}"
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)                
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                print(f'Contour {i}: x={x}, y={y}, w={w}, h={h}')
                contour = contour.squeeze()  # shape becomes (N, 2)

                contour = contour.reshape(-1, 2)

                plt.imshow(mask, cmap='gray')
                plt.plot(contour[:, 0], contour[:, 1], color='lime')  # plot x vs y
                plt.scatter([x, x+w, x+w, x], [y, y, y+h, y+h], color=['red', 'green', 'blue', 'yellow'])
                plt.title(f'{prompt[0]}')
                plt.show()

                writer.writerow([
                    prompt[1],
                    #type,
                    x, y, w, h,
                    image_name,
                    mask.shape[1],
                    mask.shape[0],
                    mask_path
                ])

##
create_dataset(train_prompts, 'train', firstRow=False)
create_dataset(valid_prompts, 'val', firstRow=False)
create_dataset(test_prompts, 'test', firstRow=False)
# %%
def copyImages(prompts_in, output_type):
    for prompt in prompts_in:
        image_name = prompt[0]
        img_path = f'{srcDir}/B-mode/{image_name}'
        dst = f'{desDir}/images/{output_type}/{image_name}'
        shutil.copy2(img_path, dst)

# %%
copyImages(train_prompts, 'train')
copyImages(valid_prompts, 'val')
copyImages(test_prompts, 'test')
# %%
