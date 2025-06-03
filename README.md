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


| Dataset               | Link                                                                                                            | Organ Type        | Added |
|-----------------------|----------------------------------------------------------------------------------------------------------------|-------------------|-------|
| 105US                 | [researchgate](https://www.researchgate.net/publication/329586355_100_2D_US_Images_and_Tumor_Segmentation_Masks) | Breast            | Yes   |
| AbdomenUS             | [kaggle](https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm)                                           | Abdomen           | No    |
| ACOUSLIC              | [grand-challenge](https://acouslic-ai.grand-challenge.org/overview-and-goals/)                                 | Liver             | No    |
| ASUS                  | [onedrive](https://onedrive.live.com/?authkey=%21AMIrL6S1cSjlo1I&id=7230D4DEC6058018%2191725&cid=7230D4DEC6058018) | Liver             | No    |
| AUL                   | [zenodo](https://zenodo.org/records/7272660)                                                                   | Lung              | No    |
| brachial plexus       | [github](https://github.com/Regional-US/brachial_plexus)                                                       | Nerve             | No    |
| BrEaST                | [cancer imaging archive](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/)                  | Breast            | Yes   |
| BUID                  | [qamebi](https://qamebi.com/breast-ultrasound-images-database/)                                                | Breast            | Yes   |
| BUS_UC                | [mendeley](https://data.mendeley.com/datasets/3ksd7w7jkx/1)                                                    | Breast            | Yes   |
| BUS_UCML              | [mendeley](https://data.mendeley.com/datasets/7fvgj4jsp7/1)                                                    | Breast            | Yes   |
| BUS-BRA               | [github](https://github.com/wgomezf/BUS-BRA)                                                                   | Breast            | Yes   |
| BUS (Dataset B)       | [mmu](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php)                                                      | Breast            | Yes   |
| BUSI                  | [HomePage](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)                                                  | Breast            | Yes   |
| CAMUS                 | [insa-lyon](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8)     | Heart             | No    |
| CardiacUDC            | [kaggle](https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset)                                | Heart             | No    |
| CCAUI                 | [mendeley](https://data.mendeley.com/datasets/d4xt63mgjm/1)                                                    | Carotid Artery    | Yes   |
| DDTI                  | [github](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN3K.md)                    | Thyroid           | No    |
| EchoCP                | [kaggle](https://www.kaggle.com/datasets/xiaoweixumedicalai/echocp)                                            | Heart             | No    |
| EchoNet-Dynamic       | [github](https://github.com/echonet/dynamic)                                                                   | Heart             | No    |
| EchoNet-Pediatric     | [github](https://echonet.github.io/pediatric)                                                                  | Heart             | No    |
| FALLMUD               | [kalisteo](https://kalisteo.cea.fr/index.php/fallmud/#)                                                        | Muscle            | No    |
| FASS                  | [mendeley](https://data.mendeley.com/datasets/4gcpm9dsc3/1)                                                    | Abdominal Organs  | No    |
| Fast-U-Net            | [github](https://github.com/vahidashkani/Fast-U-Net)                                                           | Breast            | No    |
| FH-PS-AOP             | [zenodo](https://zenodo.org/records/10829116)                                                                  | Prostate          | No    |
| GIST514-DB            | [github](https://github.com/howardchina/query2)                                                                | Liver             | No    |
| HC                    | [grand-challenge](https://hc18.grand-challenge.org/)                                                           | Head Circumference| No    |
| kidneyUS              | [github](https://github.com/rsingla92/kidneyUS)                                                                | Kidney            | No    |
| LUSS_phantom          | [Leeds](https://archive.researchdata.leeds.ac.uk/1263/)                                                        | Lung              | No    |
| MicroSeg              | [zenodo](https://zenodo.org/records/10475293)                                                                  | Prostate          | No    |
| MMOTU-2D              | [github](https://github.com/cv516Buaa/MMOTU_DS2Net)                                                            | Multi-organ       | No    |
| MMOTU-3D              | [github](https://github.com/cv516Buaa/MMOTU_DS2Net)                                                            | Multi-organ       | No    |
| MUP                   | [zenodo](https://zenodo.org/records/10475293)                                                                  | Prostate          | No    |
| regPro                | [HomePage](https://muregpro.github.io/data.html)                                                               | Prostate          | No    |
| S1                    | [ncbi](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8205136/)                                                  | Thyroid           | No    |
| Segthy                | [TUM](https://www.cs.cit.tum.de/camp/publications/segthy-dataset/)                                             | Thyroid           | No    |
| STMUS_NDA             | [mendeley](https://data.mendeley.com/datasets/3jykz7wz8d/1)                                                    | Thyroid           | No    |
| STU-Hospital          | [github](https://github.com/xbhlk/STU-Hospital)                                                                | Breast            | No    |
| TG3K                  | [github](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN3K.md)                    | Thyroid           | No    |
| Thyroid US Cineclip   | [standford](https://stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5)              | Thyroid           | No    |
| TN3K                  | [github](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN3K.md)                    | Thyroid           | No    |
| TNSCUI                | [grand-challenge](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN-SCUI2020.md)    | Thyroid           | No    |
| UPBD                  | [HomePage](https://ubpd.worldwidetracing.com:9443/)                                                            | Bladder           | No    |
| US nerve Segmentation | [kaggle](https://www.kaggle.com/c/ultrasound-nerve-segmentation/data)                                          | Nerve             | No    |

✅ = Currently added  
❌ = Not yet added

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
