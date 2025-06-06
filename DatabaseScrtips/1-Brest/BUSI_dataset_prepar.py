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
srcDir = '/home/hamze/Documents/Dataset/BreastBUSI_Images'
desDir = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/USDATASET'
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
            prompts.append(columns[1])
    return prompts
# benign_prompts = readTextPrompt(f'{desDir}/benign.csv')
# malignant_prompts = readTextPrompt(f'{desDir}/malignant.csv')
    
dataset = 'busi'
def create_dataset(type, number_list, output_type):


    with open( f'{desDir}/{output_type}_annotation.CSV', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for index in number_list:
            try:
                mask_path = f'{srcDir}/{type}/{type} ({index})_mask.png'
                image_name = f'{type} ({index}).png'
                img_path = f'{srcDir}/{type}/{image_name}'
                dst = f'{desDir}/images/{output_type}/{dataset}_{image_name}'
                shutil.copy2(img_path, dst)

                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for i, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    writer.writerow([
                        type.lower(),
                        x, y, w, h,
                        f'{dataset}_{image_name}',
                        mask.shape[1],
                        mask.shape[0],
                        mask_path,
                        dataset
                    ])
            except Exception as e:
                print(f"Error processing {index}: {str(e)}")

#%%
numbers = list(range(1, 211))
random.shuffle(numbers)

total = len(numbers)
train_size = int(0.7 * total)  # 70%
valid_size = int(0.2 * total)  # 20%
test_size = total - train_size - valid_size  # Remaining 10%

# Split the numbers
train_numbers = numbers[:train_size]
valid_numbers = numbers[train_size:train_size+valid_size]
test_numbers = numbers[train_size+valid_size:]

create_dataset('malignant', train_numbers,'train')
create_dataset('malignant', valid_numbers,'val')
create_dataset('malignant', test_numbers,'test')
#%%
numbers = list(range(1, 438))
random.shuffle(numbers)

total = len(numbers)
train_size = int(0.7 * total)  # 70%
valid_size = int(0.2 * total)  # 20%
test_size = total - train_size - valid_size  # Remaining 10%

# Split the numbers
train_numbers = numbers[:train_size]
valid_numbers = numbers[train_size:train_size+valid_size]
test_numbers = numbers[train_size+valid_size:]

create_dataset('benign', train_numbers,'train')
create_dataset('benign', valid_numbers,'val')
create_dataset('benign', test_numbers,'test')        

# %%
