#%%
import glob
datasets = ['breast','buid', 'busuc','busuclm','busb', 
            'busi','stu','s1','tn3k','tg3k',
            '105us','aul','muregpro','regpro','kidnyus']
# datasets = ['busbra', 'tnscui', 'luminous' ]
desDir = '../Grounding-Sam-Ultrasound/multimodal-data'
output_types = ['train', 'val', 'test_mask']
dataset_counts = {'train':0, 'val':0, 'test_mask':0}
print('Dataset', 'Total',output_types)
for dataset in datasets:
    row = ''
    row_total = 0
    for output_type in output_types:
        
        files = glob.glob(f'{desDir}/{output_type}/{dataset}_*')
        row += f' & {len(files)}'
        row_total+=len(files)
        dataset_counts[output_type] += len(files)
    print(dataset,'\t',row_total,row)
total = 0
for k in dataset_counts:
    print('Total',k,dataset_counts[k])
    total+=dataset_counts[k]
print('Total',total)
# %%
