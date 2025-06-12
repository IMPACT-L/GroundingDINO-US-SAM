# #%%
# import csv
# from collections import Counter
# #%%
# desDir = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data'

# dataset_counts = Counter()

# for output_type in ['test','train', 'val']:
#     file_path = f'{desDir}/{output_type}.CSV'
#     print('*'*10,output_type,'*'*10)
#     with open(file_path, 'r', newline='') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             dataset = row['dataset'].strip().lower()
#             dataset_counts[dataset] += 1

#     for dataset, count in sorted(dataset_counts.items()):
#         print(f"{dataset}: {count}")

# # %%
# #%%
# import csv
# from collections import Counter
# import os

# #%%
# desDir = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data'
# splits = ['train', 'val', 'test']
# overall_counts = Counter()

# for output_type in splits:
#     dataset_counts = Counter()  # Reset for each split
#     file_path = os.path.join(desDir, f'{output_type}.CSV')
    
#     print('*' * 10, output_type.upper(), '*' * 10)

#     with open(file_path, 'r', newline='') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             dataset = row['dataset'].strip().lower()
#             dataset_counts[dataset] += 1
#             overall_counts[dataset] += 1

#     total = sum(dataset_counts.values())
#     for dataset, count in sorted(dataset_counts.items()):
#         print(f"{dataset}: {count}")
#     print(f"Total ({output_type}): {total}\n")

# # Print grand totals across all splits
# print('*' * 10, 'TOTAL ACROSS ALL SPLITS', '*' * 10)
# grand_total = sum(overall_counts.values())
# for dataset, count in sorted(overall_counts.items()):
#     print(f"{dataset}: {count}")
# print(f"Grand Total: {grand_total}")


# %%
#%%
import glob
datasets = ['breast','buid', 'busuc','busuclm','busb', 
            'busi','stu','s1','tn3k','tg3k',
            '105us','aul','muregpro','regpro','kidnyus']
datasets = ['busbra', 'tnscui', 'luminous' ]
desDir = '/home/hamze/Documents/Grounding-Sam-Ultrasound/multimodal-data'
output_types = ['train', 'val', 'test_mask']
dataset_counts = {'train':0, 'val':0, 'test_mask':0}
print('Dataset',output_types)
for dataset in datasets:
    row = dataset+'\t'
    row_total = 0
    for output_type in output_types:
        
        files = glob.glob(f'{desDir}/{output_type}/*{dataset}*')
        row += f' & {len(files)}'
        row_total+=len(files)
        dataset_counts[output_type] += len(files)
    print(row,'&',row_total)

for k in dataset_counts:
    print('Total',k,dataset_counts[k])
# %%
