#%%
import os
methods =  ['sam2','MedSam','MedClipSam','MedClipSamv2','UniverSeg','ours']
methods =  ['UniverSeg']
# datasets = [
#     "105us", "aul", "busuclm", "stu-hospital", "s1", 
#     "busi", "busbra", "bus_uc", "bus (dataset b)", "buid", "breast",
#     "kidneyus",
#     "microseg", "regpro",
#     "tnscui", "tg3k", "tn3k"
# ]
# datasets = ["105us", "aul", "busuclm","stu","s1", "busi",
            # "busuc","busb","buid","breast","kidnyus",
            # "muregpro","regpro","tg3k","tn3k"]

datasets = ['busbra','tnscui','luminous']
for selectedDataset in datasets:
    print('*'*10,selectedDataset,'*'*10)
    for method in methods:
        save_result_path = f'visualizations/{method}/{selectedDataset}/result.txt'
        # print(save_result_path)
        if os.path.exists(save_result_path):
            with open(f'{save_result_path}', 'r') as f:
                contents = f.read().strip()  
                print(method,'->',selectedDataset)
                print(contents)
        else:
            print(method,'->',selectedDataset,'Not Exists')
            
# %%
