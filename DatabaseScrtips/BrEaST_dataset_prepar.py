#%%
from PIL import Image
import os
import cv2
import csv
import shutil
import matplotlib.pyplot as plt
import random
import matplotlib.patches as patches
random.seed(42)
#%%
srcDir = '/home/hamze/Documents/Dataset/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023'
# '/home/hamze/Documents/Dataset/BrEaST-Lesions_USG-images_and_masks-Dec-15-2023/BrEaST-Lesions_USG-images_and_masks'
desDir = '../multimodal-data/USDATASET'
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
            if columns[2] != '':
                prompts.append((columns[1],columns[20]))
            else :
                print(columns[1])
    return prompts
prompts = readTextPrompt(f'{srcDir}/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.csv')
print(prompts)
#%%
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
    with open( f'{desDir}/{output_type}_annotation.CSV', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        if firstRow:
            writer.writerow(['label_name', 'bbox_x', 'bbox_y', 
                    'bbox_width', 'bbox_height', 
                    'image_name', 'image_width', 'image_height','mask_path','dataset'])
                
        for prompt in prompts_in:
            image_name = prompt[0]
            type = prompt[1]

            mask_path = f"{srcDir}/BrEaST-Lesions_USG-images_and_masks/{image_name.replace('.png','_tumor.png')}"
            # dst = f'{desDir}/images/{output_type}/{image_name}'
            # shutil.copy2(mask_path, dst)
            print(mask_path)
            mask = cv2.imread(mask_path)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for i, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                writer.writerow([
                    # prompt,
                    type,
                    x, y, w, h,
                    image_name,
                    mask.shape[1],
                    mask.shape[0],
                    mask_path,
                    'BrEaST'
                ])

                fig, ax = plt.subplots()
                ax.imshow(mask)
                rect = patches.Rectangle((x, y), w, h,
                                        linewidth=2, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

                plt.title("Bounding Box")
                # plt.axis('off')
                plt.show()
                    
               
            # break

#%%
create_dataset(train_prompts, 'train', firstRow=False)
create_dataset(valid_prompts, 'val', firstRow=False)
create_dataset(test_prompts, 'test', firstRow=False)
# %%
def copyImages(prompts_in, output_type):
    for prompt in prompts_in:
        image_name = prompt[0]
        img_path = f'{srcDir}/BrEaST-Lesions_USG-images_and_masks/{image_name}'
        dst = f'{desDir}/images/{output_type}/{image_name}'
        shutil.copy2(img_path, dst)

# %%
copyImages(train_prompts, 'train')
copyImages(valid_prompts, 'val')
copyImages(test_prompts, 'test')
# %%
