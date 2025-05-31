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
desDir = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/USDATASET'
#%%
os.makedirs(f'{desDir}/images/train', exist_ok=True)
os.makedirs(f'{desDir}/images/val', exist_ok=True)
os.makedirs(f'{desDir}/images/test', exist_ok=True)

header = ['label_name', 'bbox_x', 'bbox_y', 
    'bbox_width', 'bbox_height', 
    'image_name', 'image_width', 'image_height','mask_path','dataset']
    
output_types = ['test','train','val']
for output_type in output_types:
    with open( f'{desDir}/{output_type}_annotation.CSV', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

# %%
