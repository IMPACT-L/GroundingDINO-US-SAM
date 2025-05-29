# Grounding DINO 


We have expanded on the original DINO  repository 
https://github.com/IDEA-Research/GroundingDINO 
by introducing the capability to train the model with image-to-text grounding. This capability is essential in applications where textual descriptions must align with regions of an image. For instance, when the model is given a caption "a cat on the sofa," it should be able to localize both the "cat" and the "sofa" in the image.

## Features:

- **Fine-tuning DINO**: This extension works allows you to fine-tune DINO on your custom dataset.
- **EMA Model**: Exponential Moving Average model to retain pre-trained knowledge
- **LORA Training** (New) : Parameter-efficient fine-tuning using LORA that trains less than 2% of parameters while maintaining performance. For example we use a rank of 32 for the LORA adapters you can aso try smaller ranks. **LoRA only saves the newly introduced parameters, significantly reducing the storage space required for fine-tuned models. During inference, these LoRA parameters are merged with the base model weights.**
* **Example Dataset**: Includes small sample dataset for training and testing
- **NMS (Optional)**: We also implemented phrase based NMS to remove redundant boxes of same objects (might be useful if you have too many detections original DETR like model which grouding dino is also based on donot require NMS)



## Installation:
See original Repo for installation of required dependencies essentially we need to install prerequisits  

```bash
pip install -r reqirements.txt
```
then install the this package
```bash
pip install -e .
```

Optional:
You might need to do this if you have an old gpu or if its arch is not recognized automatically

```bash
## See if the CUDA_HOME is set up correctly
## e.g export CUDA_HOME=/usr/local/cuda
pip uninstall groundingdino
nvidia-smi --query-gpu=gpu_name,compute_cap --format=csv
export TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6" (add your gpu arch given from previous)
export FORCE_CUDA=1
pip install -e .

```


## Dataset

| Dataset               | Link                                                                                                            | Added |
|-----------------------|----------------------------------------------------------------------------------------------------------------|-------|
| 105US                 | [researchgate](https://www.researchgate.net/publication/329586355_100_2D_US_Images_and_Tumor_Segmentation_Masks) | No    |
| AbdomenUS             | [kaggle](https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm)                                           | No    |
| ACOUSLIC              | [grand-challenge](https://acouslic-ai.grand-challenge.org/overview-and-goals/)                                 | No    |
| ASUS                  | [onedrive](https://onedrive.live.com/?authkey=%21AMIrL6S1cSjlo1I&id=7230D4DEC6058018%2191725&cid=7230D4DEC6058018) | No    |
| AUL                   | [zenodo](https://zenodo.org/records/7272660)                                                                   | No    |
| brachial plexus       | [github](https://github.com/Regional-US/brachial_plexus)                                                       | No    |
| BrEaST                | [cancer imaging archive](https://www.cancerimagingarchive.net/collection/breast-lesions-usg/)                  | No    |
| BUID                  | [qamebi](https://qamebi.com/breast-ultrasound-images-database/)                                                | No    |
| BUS_UC                | [mendeley](https://data.mendeley.com/datasets/3ksd7w7jkx/1)                                                    | No    |
| BUS_UCML              | [mendeley](https://data.mendeley.com/datasets/7fvgj4jsp7/1)                                                    | Yes   |
| BUS-BRA               | [github](https://github.com/wgomezf/BUS-BRA)                                                                   | Yes   |
| BUS (Dataset B)       | [mmu](http://www2.docm.mmu.ac.uk/STAFF/M.Yap/dataset.php)                                                      | Yes   |
| BUSI                  | [HomePage](https://scholar.cu.edu.eg/?q=afahmy/pages/dataset)                                                  | Yes   |
| CAMUS                 | [insa-lyon](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/6373703d73e9f0047faa1bc8)     | No    |
| CardiacUDC            | [kaggle](https://www.kaggle.com/datasets/xiaoweixumedicalai/cardiacudc-dataset)                                | No    |
| CCAUI                 | [mendeley](https://data.mendeley.com/datasets/d4xt63mgjm/1)                                                    | No    |
| DDTI                  | [github](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN3K.md)                    | No    |
| EchoCP                | [kaggle](https://www.kaggle.com/datasets/xiaoweixumedicalai/echocp)                                            | No    |
| EchoNet-Dynamic       | [github](https://github.com/echonet/dynamic)                                                                   | No    |
| EchoNet-Pediatric     | [github](https://echonet.github.io/pediatric)                                                                  | No    |
| FALLMUD               | [kalisteo](https://kalisteo.cea.fr/index.php/fallmud/#)                                                        | No    |
| FASS                  | [mendeley](https://data.mendeley.com/datasets/4gcpm9dsc3/1)                                                    | No    |
| Fast-U-Net            | [github](https://github.com/vahidashkani/Fast-U-Net)                                                           | No    |
| FH-PS-AOP             | [zenodo](https://zenodo.org/records/10829116)                                                                  | No    |
| GIST514-DB            | [github](https://github.com/howardchina/query2)                                                                | No    |
| HC                    | [grand-challenge](https://hc18.grand-challenge.org/)                                                           | No    |
| kidneyUS              | [github](https://github.com/rsingla92/kidneyUS)                                                                | No    |
| LUSS_phantom          | [Leeds](https://archive.researchdata.leeds.ac.uk/1263/)                                                        | No    |
| MicroSeg              | [zenodo](https://zenodo.org/records/10475293)                                                                  | No    |
| MMOTU-2D              | [github](https://github.com/cv516Buaa/MMOTU_DS2Net)                                                            | No    |
| MMOTU-3D              | [github](https://github.com/cv516Buaa/MMOTU_DS2Net)                                                            | No    |
| MUP                   | [zenodo](https://zenodo.org/records/10475293)                                                                  | No    |
| regPro                | [HomePage](https://muregpro.github.io/data.html)                                                               | No    |
| S1                    | [ncbi](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8205136/)                                                  | No    |
| Segthy                | [TUM](https://www.cs.cit.tum.de/camp/publications/segthy-dataset/)                                             | No    |
| STMUS_NDA             | [mendeley](https://data.mendeley.com/datasets/3jykz7wz8d/1)                                                    | No    |
| STU-Hospital          | [github](https://github.com/xbhlk/STU-Hospital)                                                                | No    |
| TG3K                  | [github](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN3K.md)                    | No    |
| Thyroid US Cineclip   | [standford](https://stanfordaimi.azurewebsites.net/datasets/a72f2b02-7b53-4c5d-963c-d7253220bfd5)              | No    |
| TN3K                  | [github](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN3K.md)                    | No    |
| TNSCUI                | [grand-challenge](https://github.com/openmedlab/Awesome-Medical-Dataset/blob/main/resources/TN-SCUI2020.md)    | No    |
| UPBD                  | [HomePage](https://ubpd.worldwidetracing.com:9443/)                                                            | No    |
| US nerve Segmentation | [kaggle](https://www.kaggle.com/c/ultrasound-nerve-segmentation/data)                                          | No    |

## Train: 

1. Prepare your dataset with images and associated textual captions. A tiny dataset is given multimodal-data to demonstrate the expected data format.
2. Run the train.py for training. We use a batch size of 8, a learning rate of 1e-5, and the AdamW optimizer. The model is trained for 100 epochs. **See `configs/train_config.yaml` for detailed training configurations.**

  ```
  python train.py
  ```

## Test:
Visualize results of training on test images. **See `configs/test_config.yaml` for detailed testing configurations.**

```
python test.py
```


## Qualitative Results

For Input text "shirt. pants. bag" and input validation images (see above like for train and valiadtion data. The model was only trained on 200 images and tested on 50 images) 


**Before Fine-tuning**: Model performs as shown on left below. GT is shown in green and model predictions are shown in red. Interesting to note is that for this dataset model does not perform very bad, but the concept of some categories is different e.g "shirt" is different then the GT see second and third image. 

**After Fine-tuning**: Model correctly detects all required categories image one along with the correct concept.


TO DO:

