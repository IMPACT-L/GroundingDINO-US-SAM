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
srcDir = '/home/hamze/Documents/Dataset/BUSBRA'
desDir = '../multimodal-data/USDATASET'
dataset = 'busuc'
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
            prompts.append((columns[0],columns[3],columns[6],columns[7],columns[9]))
    return prompts
prompts = readTextPrompt(f'{srcDir}/bus_data.csv')
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
                
        for prompt in prompts_in:
            image_name = f'{prompt[0]}.png'
            type = prompt[1]
            bbox = list(map(int, prompt[4].strip('[]').split(',')))
            x, y, w, h = bbox
            # width = x_max - x_min
            # height = y_max - y_min

            mask_path = f"{srcDir}/Masks/{image_name.replace('bus','mask')}"
            print(mask_path)
            # img_path = f'{srcDir}/original/{image_name}'
            # mask = cv2.imread(mask_path)
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

            # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # fig, ax = plt.subplots()
            # ax.imshow(mask)
            # rect = patches.Rectangle((x, y), w, h,
            #                         linewidth=2, edgecolor='r', facecolor='none')
            # ax.add_patch(rect)

            # plt.title("Bounding Box")
            # # plt.axis('off')
            # plt.show()
                
            # print([
            #       type,
            #     x, y, w, h,
            #     image_name,
            #     mask.shape[1],
            #     mask.shape[0],
            #     mask_path,
            # ])

            writer.writerow([
                type.lower(),
                x, y, w, h,
                f'{dataset}_{image_name}',
                prompt[2],
                prompt[3],
                mask_path,
                dataset
            ])
            # break

#%%
create_dataset(train_prompts, 'train', firstRow=False)
create_dataset(valid_prompts, 'val', firstRow=False)
create_dataset(test_prompts, 'test', firstRow=False)
# %%
def copyImages(prompts_in, output_type):
    for prompt in prompts_in:
        image_name = f'{prompt[0]}.png'
        img_path = f'{srcDir}/Images/{image_name}'
        dst = f'{desDir}/images/{output_type}/{dataset}_{image_name}'
        shutil.copy2(img_path, dst)

# %%
copyImages(train_prompts, 'train')
copyImages(valid_prompts, 'val')
copyImages(test_prompts, 'test')
# %%
