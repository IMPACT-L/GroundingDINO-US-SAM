#%%
import os
methods =  {'UniverSeg':'UniverSeg',
            'BiomedParse':'BiomedParse',
            'SAMUS':'SAMUS',
            'MedClipSam':'MedClip-SAM',
            'MedClipSamv2':'MedClip-SAMv2',
            'ours':'Ours'} # Abilation Compares

datasets = [{"Breast":{"breast":"BrEaST", "buid":"BUID", "busuc":"BUSUC", "busuclm":"BUSUCLM","busb":"BUSB","busi":"BUSI","stu":"STU","s1":"S1"}},
            {"Thyroid":{"tn3k":"TN3K","tg3k":"TG3K"}},
            {"Liver":{"105us":"105US","aul":"AUL"}},
            {"Prostate":{"muregpro":"MicroSeg","regpro":"RegPro"}},
            {"Kidney":{"kidnyus":"KidneyUS"}}
            ]
            
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
