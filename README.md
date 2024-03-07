# Align_CBM

(For MICCAI submission) A concept bottleneck model based framework with medical knowledge integration by loss_align.

## Medical knowledge representation
- WBC datasets: `dataset_txt/pbc_alpha_true_11.csv`

- Skin datasets: `skincon/dataset_txt/skincon_alpha_true_v3.csv`

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
    ```

## Experiment

Train and test the CBM with `loss_align`:
  
  `python train_test.py --lamda 1 --backbone vgg16 --classifier linear`

- `--lamda 0` is the baseline without loss_align, `lamda` can be larger than 1. 
