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
srcDir = '/home/hamze/Documents/Dataset/BUS-UCLM'
desDir = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/USDATASET'
#%%
os.makedirs(f'{desDir}/images/train', exist_ok=True)
os.makedirs(f'{desDir}/images/val', exist_ok=True)
os.makedirs(f'{desDir}/images/test', exist_ok=True)
#%%
mask_paths = glob.glob(f'{srcDir}/masks/*')
types = ['normal','benign','malignant']
benigins = []
malignants = []
for mask_path in mask_paths:
    mask = cv2.imread(mask_path) 
    file_name = mask_path.split('/')[-1]
    new_mask_path = f'{srcDir}/masks_new/{file_name}'
    
    for i in [1,2]:
        channel = mask[:, :, i]
        if channel.any():
            contours, _ = cv2.findContours(channel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for _, contour in enumerate(contours):
                x, y, w, h = cv2.boundingRect(contour)
                row =[
                    types[i],
                    x, y, w, h,
                    file_name,
                    mask.shape[1],
                    mask.shape[0],
                    new_mask_path,
                    'BUS-UCLM'
                ]
                binary_mask = (channel > 0).astype(np.uint8)  # 0 and 1
                save_mask = (binary_mask * 255).astype(np.uint8)  # Scale to 0–255 for saving

                cv2.imwrite(new_mask_path, save_mask)
                malignants.append(row) if i==2 else benigins.append(row)
                # shutil.copy2(mask_path, new_mask_path)
            break
        
    print(types[i])

# plt.imshow(mask)
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
                img_path = f'{srcDir}/images/{image_name}'
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

create_dataset('benign', train_numbers,'train',False)
create_dataset('benign', valid_numbers,'val',False)
create_dataset('benign', test_numbers,'test',False)

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
create_dataset('malignant', train_numbers,'train',False)
create_dataset('malignant', valid_numbers,'val',False)
create_dataset('malignant', test_numbers,'test',False)
# %%
