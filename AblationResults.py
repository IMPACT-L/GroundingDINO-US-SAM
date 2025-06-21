#%%
import os
methods =  {'sam2':'Grounding DINO + SAM2',
            'MedSam':'Grounding DINO + SAM2',
            'ours':'Grounding DINO + SAM2'} # Abilation Compares

datasets = [{"Breast":{"breast":"BrEaST", "buid":"BUID", "busi":"BUSI"}},
            {"Liver":{"aul":"AUL"}},
            {"Prostate":{"muregpro":"MicroSeg"}}]

for organs in datasets:
    # print(organs)
    for organ in organs:
        print('\midrule')
        print("\multirow{"+str(len(organs[organ]))+"}{*}{"+organ+"} ")
        # print('& ',organ,'\t & ')
        row=''
        for key in organs[organ]:
            row = f'& {organs[organ][key]}'

            for method in methods:
                # print(method)
                dataset_path = f'visualizations/{method}/{key}/result.txt'
                # print(dataset_path)
                if os.path.exists(dataset_path):
                    with open(f'{dataset_path}', 'r') as f:
                        contents = f.read().strip()  
                        row+=f"\t& {contents.split(':')[-1]}"
                        # print(method,'->',selectedDataset)
                        # print(contents.split(':')[-1])
                else:
                    print(method,'->',key,'Not Exists')
            print(row+'\\\\')
# %%
