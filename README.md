# Align_CBM

(For MICCAI submission) A concept bottleneck model based framework with medical knowledge integration by loss_align.

## Medical knowledge representation
- WBC datasets: [`dataset_txt/pbc_alpha_true_11.csv`](https://github.com/PangWinnie0219/align_concept_cbm/blob/main/dataset_txt/pbc_alpha_true_11.csv)

- Skin datasets: [`skincon/dataset_txt/skincon_alpha_true_v3.csv`](https://github.com/PangWinnie0219/align_concept_cbm/blob/main/skincon/dataset_txt/skincon_alpha_true_v3.csv)

## Data preparation
Prepare a `./data` directory that contains all images of the datasets:

    ```
    - data
        - PBC
            - PBC_dataset_normal_DIB
        - RaabinWBC
            - TestA
        - scirep
            - test_crop
        - fitz_images
        - ddi_images
    ```
You may download the images from [PBC dataset](https://data.mendeley.com/datasets/snkd93bnjr/1), [RaabinWBC](https://raabindata.com/free-data/), [scirep](https://www.nature.com/articles/s41598-023-29331-3), [Fitzpatrick 17k](https://github.com/mattgroh/fitzpatrick17k), [DDI](https://ddi-dataset.github.io/index.html#paper) and concept annotations from [WBCAtt](https://rose1.ntu.edu.sg/dataset/WBCAtt/), [SkinCon](https://skincon-dataset.github.io/index.html#dataset).  

## Experiment

Train and test the CBM with `loss_align`:
  
  `python train_test.py --lamda 1 --backbone vgg16 --classifier linear`

- `--lamda 0` is the baseline without loss_align, `lamda` can be larger than 1. 
