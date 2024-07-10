# Integrating Clinical Knowledge into CBMs

This is the Pytorch implementation for our paper:

 - Winnie Pang, Xueyi Ke, Satoshi Tsutsui, and Bihan Wen. (2024). Integrating Clinical Knowledge into Concept Bottleneck Models. International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI). 


![overview](https://github.com/PangWinnie0219/align_concept_cbm/blob/main/figs/overview.png)

We propose a method to guide concept bottleneck models (CBMs) using knowledge aligned with clinicians' perspectives. 

(a): CBMs predict interpretable concepts (e.g., granule color, cell shape, etc.) and then make a final prediction (e.g., eosinophil) based on them. During training, models usually do not consider the clinical importance of the concepts. Therefore, granule color and cell shape are treated equally despite granule color being a much more important factor for predicting eosinophil. 

(b): To incorporate clinical knowledge, we enforce the CBM to exhibit a significant drop in cell type prediction probabilities when a clinically important concept is removed from the prediction. For instance, the predicted eosinophil probability should be lower when granule color, a key factor in recognizing eosinophil, is missing. 

(c): Conversely, the cell type prediction probabilities should experience a negligible drop when a less clinically important concept is removed from the prediction. For instance, the eosinophil probability should not be affected much when cell shape, which is irrelevant to recognizing eosinophil, is missing.

## Environment

Install the packages required using the requirements.txt file:

 `pip install -r requirements.txt `

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

-  Check `python train_test.py --help` for arguments.
- `--lamda 0` is the baseline without loss_align, `lamda` can be larger than 1. 

## Acknowledgement

Some parts of the code are adopted from [WBCAtt](https://github.com/apple2373/wbcatt). 


## Citation

If you find our code or paper useful, please cite as:

```
@inproceedings{pang2024integrating,
  title={Integrating Clinical Knowledge into Concept Bottleneck Models},
  author={Pang, Winnie and Ke, Xueyi and Tsutsui, Satoshi and Wen, Bihan},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  pages={},
  year={2024},
  organization={}
}
```
