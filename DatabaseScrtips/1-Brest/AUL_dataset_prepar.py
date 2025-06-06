# #%%
# from PIL import Image
# import os
# import cv2
# import csv
# import shutil
# import matplotlib.pyplot as plt
# import random
# import glob
# import numpy as np
# import json
# import matplotlib.patches as patches

# random.seed(42)
# #%%
# srcDir = '/home/hamze/Documents/Dataset/AUL'
# desDir = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data/USDATASET'
# #%%
# os.makedirs(f'{desDir}/images/train', exist_ok=True)
# os.makedirs(f'{desDir}/images/val', exist_ok=True)
# os.makedirs(f'{desDir}/images/test', exist_ok=True)
# dataset = 'aul'
# #%%

# def create_dataset(type): 
#     list = []
#     mask_paths = glob.glob(f'{srcDir}/{type}/segmentation/mass/*')
#     for mask_path in mask_paths:
    
#         img_path = mask_path.replace('segmentation/mass','image').replace('json','jpg')
#         mask = cv2.imread(img_path)
#         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

#         with open(mask_path, 'r') as file:
#             points = json.load(file)
#         points = np.array(points)

#         # print(points)
#         x_min = np.min(points[:, 0])
#         x_max = np.max(points[:, 0])
#         y_min = np.min(points[:, 1])
#         y_max = np.max(points[:, 1])

#         bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
#         row = [
#             type.lower(),
#             x_min, y_min, x_max - x_min, y_max - y_min
#             image_name,
#             mask.shape[1],
#             mask.shape[0],
#             mask_path,
#             dataset
#         ]
#         fig, ax = plt.subplots(figsize=(8, 6))
#         ax.imshow(mask)
#         rect = patches.Rectangle((x_min, y_min), bbox[2], bbox[3],
#                                 linewidth=2, edgecolor='red', facecolor='none')
#         ax.add_patch(rect)
#         ax.set_title('Points and Bounding Box')
#         ax.legend()
#         plt.gca().invert_yaxis()
#         plt.show()

# create_dataset('Malignant')
# #%%
# mask_paths = glob.glob(f'{srcDir}/Malignant/masks/*')
# malignants = []
# for mask_path in mask_paths:
#     mask = cv2.imread(mask_path)
#     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#     image_name = mask_path.split('/')[-1]
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for _, contour in enumerate(contours):
#         x, y, w, h = cv2.boundingRect(contour)
#         row =[
#             'malignant',
#             x, y, w, h,
#             image_name,
#             mask.shape[1],
#             mask.shape[0],
#             mask_path,
#             dataset
#         ]
       
#         malignants.append(row)
#         # shutil.copy2(mask_path, new_mask_path)
#     # break
# print(malignants)
# # os.rename(benigin_mask_path, benigin_mask_path.replace('.png','_bus_us.png'))
# #%%
# mask_paths = glob.glob(f'{srcDir}/Benign/masks/*')
# benigins = []
# for mask_path in mask_paths:
#     mask = cv2.imread(mask_path)
#     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
#     image_name = mask_path.split('/')[-1]
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for _, contour in enumerate(contours):
#         x, y, w, h = cv2.boundingRect(contour)
#         row =[
#             'benigin',
#             x, y, w, h,
#             image_name,
#             mask.shape[1],
#             mask.shape[0],
#             mask_path,
#             dataset
#         ]
       
#         benigins.append(row)
#         # shutil.copy2(mask_path, new_mask_path)
#     # break
# print(benigins)
# #%%
# def create_dataset(type, number_list, output_type):
#     with open( f'{desDir}/{output_type}_annotation.CSV', 'a', newline='') as csvfile:
#         writer = csv.writer(csvfile)

#         for row in number_list:
#             try:
#                 writer.writerow(row)
#                 image_name = row[5]
#                 img_path = f'{srcDir}/{type}/images/{image_name}'
#                 dst = f'{desDir}/images/{output_type}/{image_name}'
                
#                 shutil.copy2(img_path, dst)
#                 print('saved')
               
#             except Exception as e:
#                 print(f"Error processing {row}: {str(e)}")

# #%%
# random.shuffle(benigins)

# total = len(benigins)
# train_size = int(0.7 * total)  # 70%
# valid_size = int(0.2 * total)  # 20%
# test_size = total - train_size - valid_size  # Remaining 10%

# # Split the numbers
# train_numbers = benigins[:train_size]
# valid_numbers = benigins[train_size:train_size+valid_size]
# test_numbers = benigins[train_size+valid_size:]
# print(len(train_numbers),len(valid_numbers),len(test_numbers))

# create_dataset('Benign', train_numbers,'train')
# create_dataset('Benign', valid_numbers,'val')
# create_dataset('Benign', test_numbers,'test')

# #%%
# random.shuffle(malignants)

# total = len(malignants)
# train_size = int(0.7 * total)  # 70%
# valid_size = int(0.2 * total)  # 20%
# test_size = total - train_size - valid_size  # Remaining 10%

# # Split the numbers
# train_numbers = malignants[:train_size]
# valid_numbers = malignants[train_size:train_size+valid_size]
# test_numbers = malignants[train_size+valid_size:]
# print(len(train_numbers),len(valid_numbers),len(test_numbers))
# create_dataset('Malignant', train_numbers,'train')
# create_dataset('Malignant', valid_numbers,'val')
# create_dataset('Malignant', test_numbers,'test')
# # %%
