#%%
import numpy as np
from scipy.stats import ttest_ind,ttest_rel
#%%
def pvalue(x1,x2):
    t_statistic, p_value = ttest_rel(x1, x2)
    return p_value
# #%%
# datasets = ["busbra","tnscui","luminous"]
datasets = ["breast", "buid", "busuc","busuclm","busb", "busi",
                "stu","s1","tn3k","tg3k","105us",
                "aul","muregpro","regpro","kidnyus"]
# dataset = "busbra"

# datasets = ["busbra","tnscui","breast", "buid", "busuc","busuclm","busb", "busi",
#                 "stu","s1","tn3k","tg3k","105us",
#                 "aul","muregpro","regpro","kidnyus"]

# datasets = ["breast","buid","busi", "aul", "muregpro"]

# %%
# target = ["sam2","MedSam","UniverSeg","BiomedParse","SAMUS", "MedCLIP-SAM" "MedCLIP-SAMv2", "Ours"]
targets = ["UniverSeg","BiomedParse","SAMUS", "MedClipSam", "MedClipSamv2"]
# type = "dices" "ious"

# print('*'*10,target,type,'*'*10,)
# targets = [ "MedClipSamv2"]

dataset = "luminous"
for target in targets:
    path_ours = f'visualizations/ours/{dataset}'
    
    ious_ours = np.loadtxt(f'{path_ours}/ious.txt')
    ious = np.loadtxt(f'visualizations/{target}/{dataset}/ious.txt')

    dices_ours = np.loadtxt(f'{path_ours}/dices.txt')
    dices = np.loadtxt(f'visualizations/{target}/{dataset}/dices.txt')

    length = min(len(ious_ours),len(ious))


    p_value_ious = pvalue(ious_ours[:length],ious[:length])
    starts_ious = ""
    if p_value_ious < 0.005:
        starts_ious="**"
    elif p_value_ious < 0.01:
        starts_ious="*"
    else:
        starts_ious="NS"

    p_value_dices = pvalue(dices_ours[:length],dices[:length])
    starts_dice = ""
    if p_value_dices < 0.005:
        starts_dice="**"
    elif p_value_dices < 0.01:
        starts_dice="*"
    else:
        starts_dice="NS"

    print(f"{dataset}-{target}: DCS:{p_value_dices:.5f} {starts_dice}, IOU:{p_value_ious:.5f} {starts_ious}")  
# %%
