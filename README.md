# GroundingDINO-US-SAM

**[IMPACT Lab](https://users.encs.concordia.ca/~impact/)** 

[[`Paper`](https://arxiv.org/abs/6579543)]

**Paper Title:**  
📄 *GroundingDINO-US-SAM: Text-Prompted Multi-Organ Segmentation in Ultrasound with LoRA-Tuned Vision–Language Models*

---
## Overview

## 🧠 Abstract

Accurate and generalizable object segmentation in ultrasound imaging remains a significant challenge due to anatomical variability, diverse imaging protocols, and limited annotated data. In this study, we propose a prompt-driven vision-language model (VLM) that integrates **Grounding DINO** with **SAM2** to enable object segmentation across multiple ultrasound organs.

A total of 18 public ultrasound datasets—covering breast, thyroid, liver, prostate, kidney, and paraspinal muscle—were utilized. These datasets were split into 15 for fine-tuning and validation of Grounding DINO using **Low Rank Adaptation (LoRA)**, and 3 were held out entirely for testing to evaluate generalization to unseen distributions.

Comprehensive experiments demonstrate that our method **outperforms** state-of-the-art segmentation baselines, including **UniverSeg**, **MedSAM**, **MedCLIP-SAM**, **BiomedParse**, and **SAMUS** on most seen datasets, while also maintaining strong performance on **unseen** datasets without further fine-tuning.

These findings highlight the potential of VLMs in **scalable**, **robust**, and **automated** ultrasound image analysis, reducing reliance on large, organ-specific annotated datasets.

> 🔓 *We will publish our code at* [`code.sonography.ai`](https://code.sonography.ai) *after acceptance.*

---

### Framework

<p float="left">
  <img src="assets/model.png" width="100%" />
</p>

### Sample Segmentation Results on Seen Dataset
<p float="left">
  <img src="assets/seen.png" width="100%" />
</p>

### Sample Segmentation Results on Unseen Dataset
<p float="left">
  <img src="assets/un_seen.png" width="100%" />
</p>

## 🗂 Datasets

This study utilized **18 public ultrasound datasets** spanning a wide range of anatomical targets: **breast, thyroid, liver, prostate, kidney**, and **paraspinal muscle**. These were divided into:

- **15 datasets** for fine-tuning and validation of Grounding DINO
- **3 held-out datasets** (highlighted below) for **testing on unseen domains** (no exposure during training or validation)

While most baselines used the same training and test splits, **UniverSeg** required a small **16-image support set** even for unseen datasets. Therefore, we provided 16 annotated images from each unseen dataset to ensure fair comparison.

### Summary of Dataset Splits

| Organ         | Dataset        | Total Images | Train | Val  | Test | Notes |
|---------------|----------------|--------------|-------|------|------|-------|
| Breast        | BrEaST         | 252          | 176   | 50   | 26   |       |
|               | BUID           | 233          | 161   | 46   | 26   |       |
|               | BUSUC          | 810          | 566   | 161  | 83   |       |
|               | BUSUCML        | 264          | 183   | 52   | 29   |       |
|               | BUSB           | 163          | 114   | 32   | 17   |       |
|               | BUSI           | 657          | 456   | 132  | 69   |       |
|               | STU            | 42           | 29    | 8    | 5    |       |
|               | S1             | 202          | 140   | 40   | 22   |       |
|               | **BUSBRA**     | **1875**     | ---   | ---  | 1875 | 🧪 *Unseen* |
| Thyroid       | TN3K           | 3493         | 2442  | 703  | 348  |       |
|               | TG3K           | 3565         | 2497  | 713  | 355  |       |
|               | **TNSCUI**     | **637**      | ---   | ---  | 637  | 🧪 *Unseen* |
| Liver         | 105US          | 105          | 73    | 21   | 11   |       |
|               | AUL            | 533          | 351   | 120  | 62   |       |
| Prostate      | MicroSeg       | 2283         | 1527  | 495  | 261  |       |
|               | RegPro         | 4218         | 2952  | 843  | 423  |       |
| Kidney        | kidneyUS       | 1963         | 1257  | 465  | 241  |       |
| Back Muscle   | **Luminous**   | **296**      | ---   | ---  | 296  | 🧪 *Unseen* |

**🔢 Total:**  
- 18 datasets  
- **18,783** images  
- **12,924** train / **3,881** val / **1,978** test

---


