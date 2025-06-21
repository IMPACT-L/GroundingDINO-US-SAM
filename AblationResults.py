#%%
import os
# methods =  ['Grounding DINO + SAM2','Grounding DINO + SAM2','Grounding DINO + SAM2 (Ours)'] # Abilation Compares
methods =  {'sam2':'Grounding DINO + SAM2',
            'MedSam':'Grounding DINO + SAM2',
            'ours':'Grounding DINO + SAM2'} # Abilation Compares

datasets = [{"Breast":{"breast":"BrEaST", "buid":"BUID", "busi":"BUSI"}},
            {"Liver":{"aul":"AUL"}},
            {"Prostate":{"muregpro":"MicroSeg"}}]

# %%
# \multirow{3}{*}{Breast} 
#             & BrEaST     & 19.18±25 & 13.77±23 & 50.07±27 & 37.62±23 & \textbf{74.35±24} & \textbf{63.41±22} \\
#             & BUID       & 43.47±34 & 35.33±34 & 62.18±28 & 50.22±25 & \textbf{85.64±16} & \textbf{77.73±19} \\
#             & BUSI       & 28.71±32 & 22.41±29 & 49.06±31 & 37.88±26 & \textbf{74.07±29} & \textbf{65.67±29} \\
#         \midrule
#         \multirow{1}{*}{Liver} 
#             & AUL        & 13.76±12 & 7.93±8  & 30.33±27 & 21.52±22 & \textbf{41.09±35} & \textbf{32.77±31} \\
#         \midrule
#         \multirow{1}{*}{Prostate} 
#             & MicroSeg   & 61.39±21 & 47.41±20 & 58.31±21 & 43.99±19 & \textbf{88.66±12} & \textbf{81.26±15} \\
#%%
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
