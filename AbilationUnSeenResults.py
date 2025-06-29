#%%
import os
datasets = ['busbra','tnscui','luminous']# Unseen Dataset

methods =  {'UniverSeg':'UniverSeg',
            'BiomedParse':'BiomedParse',
            'SAMUS':'SAMUS',
            'MedClipSam':'MedClip-SAM',
            'MedClipSamv2':'MedClip-SAMv2',
            'ours':'Ours'} # Abilation Compares

datasets = [
            {"Breast":{"busbra":"BUSBRA"}},
            {"Thyroid":{"tnscui":"TNSCUI"}},
            {"Back Muscle":{"luminous":"Luminous"}}
            ]
            
# %%
for organs in datasets:
    # print(organs)
    for organ in organs:
        print('\midrule')
        print("\multirow{"+str(len(organs[organ]))+"}{*}{"+organ+"} ")
        row=''
        for key in organs[organ]:
            row = f'& {organs[organ][key]}'

            for method in methods:
                dataset_path = f'visualizations/{method}/{key}/result.txt'
                if os.path.exists(dataset_path):
                    with open(f'{dataset_path}', 'r') as f:
                        contents = f.read().strip()  
                        row+=f"\t& {contents.split(':')[-1]}"
                else:
                    print(method,'->',key,'Not Exists')
            print(row+'\\\\')
# %%
