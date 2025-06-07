#!/bin/bash

root="/home/hamze/Documents/Grounding-Sam-Ultrasound"

cd "$root/DatabaseScrtips/1-Brest" || { echo "Directory not found"; exit 1; }
echo ' ************ Add Breast dataset ************ '
python BrEaST_dataset_prepar.py
python BUID_dataset_prepar.py
python BUS_B_dataset_prepar.py
python BUS_UC_dataset_prepar.py
python BUSBRA_dataset_prepar.py
python BUSI_dataset_prepar.py
python S1_dataset_prepar.py
python STU_dataset_prepar.py
python UCLM_dataset_prepar.py

cd "$root/DatabaseScrtips/5-Liver" || { echo "Directory not found"; exit 1; }
echo ' ************ Add Liver dataset ************ '
python AUL_dataset_prepar.py
python 105US_dataset_prepar.py

