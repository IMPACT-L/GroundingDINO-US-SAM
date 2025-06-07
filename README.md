# Grounding SAM for Breast and Thyroid Ultrasound Image Segmentation

Welcome to the official repository of our research on **Grounding-DINO + SAM2** for ultrasound (US) image segmentation. This project focuses on **fine-tuning GroundingDINO** for ultrasound object grounding and applying **SAM2 for segmentation mask generation**, with additional post-processing to enhance the quality of results.

## 🔍 Project Objective

Our goal is to segment and localize key structures (e.g., **tumor**, **benign**, **malignant**, **carotid artery**) in **ultrasound images**, primarily of the **breast** and **thyroid**, using state-of-the-art vision-language models and promptable segmentation frameworks.

## 🧪 Methodology

We build upon the **GroundingDINO** model for object detection and grounding using text prompts and apply **SAM2** (Segment Anything Model) for pixel-level segmentation.

### 🔧 Fine-Tuning

We fine-tune **only GroundingDINO** on US datasets. The SAM2 model is used as a frozen module for mask generation. In future work, we plan to explore:
- Fine-tuning **SAM2**
- Leveraging **contrastive learning** for better image-to/from-text embeddings (based on Taha’s method)

### 🔁 Post-processing

Since SAM's raw masks may contain **holes and cracks**, we apply:
- **Connected Components Analysis**
- **Morphological Operations**
- **Dilation**
to generate smooth and complete masks.

## 🧪 Ablation Studies

We explore the optimal configuration for:
- Fine-tuning **hyperparameters** of GroundingDINO
- Effective **post-processing** pipelines

## 📊 Datasets

We compiled a wide range of publicly available ultrasound datasets. Currently, most datasets are **breast US**. To enhance generalizability and avoid a breast-only limitation, we aim to include **thyroid** and other organ datasets.


| Dataset               | Link                                                                                                               | Organ Type         | Added |
|-----------------------|--------------------------------------------------------------------------------------------------------------------|--------------------|-------|
| BrEaST                | [Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/)                     | Breast             | ✅    |
| BUID                  | [QAMeBi](https://qamebi.com/breast-ultrasound-images-database/)                                                   | Breast             | ✅    |
| BUS_UC                | [Mendeley](https://data.mendeley.com/datasets/3ksd7w7jkx/1)                                                       | Breast             | ✅    |
| BUS_UCML              | [Mendeley](https://data.mendeley.com/datasets/7fvgj4jsp7/1)                                                       | Breast             | ✅    |
| BUS-BRA               | [GitHub](https://github.com/wgomezf/BUS-BRA)                                                                      | Breast             | ✅    |
| BUS (Dataset B)       | [MMU](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php)                                                         | Breast             | ✅    |
| BUSI                  | [Cairo University](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)                                             | Breast             | ✅    |
| STU-Hospital          | [GitHub](https://github.com/xbhlk/STU-Hospital)                                                                   | Breast             | ✅    |
| S1                    | [NCBI](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8205136/)                                                     | Breast             | ✅    |
| ————————              | ————————— | ————————             | ——    |
| TN3K                  | [GitHub](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN3K.md)                       | Thyroid            | ✅    |
| TG3K                  | [GitHub](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation)                                  | Thyroid            | ✅    |
| TNSCUI                | [GitHub](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN-SCUI2020.md)                | Thyroid            | ✅    |
| ————————              | ————————— | ————————             | ——    |
| 105US                 | [ResearchGate](https://www.researchgate.net/publication/329586355_100_2D_US_Images_and_Tumor_Segmentation_Masks)  | Liver              | ✅    |
| AUL                   | [Zenodo](https://zenodo.org/records/7272660)                                                                      | Liver              | ❌    |
| ————————              | ————————— | ————————             | ——    |
| MicroSeg              | [Zenodo](https://zenodo.org/records/10475293)                                                                     | Prostate           | DL    |
| regPro                | [Homepage](https://muregpro.github.io/data.html)                                                                  | Prostate           | DL    |
| ————————              | ————————— | ————————             | ——    |
| KiTS23                | [GitHub](https://github.com/neheller/kits23/tree/main)                                                            | Kidney             | not us|
| kidneyUS              | [GitHub](https://github.com/rsingla92/kidneyUS)                                                                   | Kidney             | wt    |
| ————————              | ————————— | ————————             | ——    |





> We plan to hold out some datasets entirely as **unseen test datasets** to evaluate generalization. This is in contrast to traditional train/val/test splits.

## 📌 Contributions

- [x] GroundingDINO fine-tuning for US grounding
- [x] Post-processing pipeline for SAM masks
- [ ] Contrastive learning integration
- [ ] Multiorgan evaluation (breast + thyroid)

## 💡 Future Work

- Add and evaluate **Thyroid** and **Carotid** datasets
- Explore **text-prompt variability**
- Incorporate **SAM2 fine-tuning**
- Benchmark against other segmentation methods
