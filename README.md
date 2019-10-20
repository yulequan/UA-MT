# Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation
by [Lequan Yu](http://yulequan.github.io), [Shujun Wang](https://emmaw8.github.io/), [Xiaomeng Li](https://xmengli999.github.io/), [Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

### Introduction

This repository is for our MICCAI 2019 paper '[Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation](https://arxiv.org/abs/1907.07034)'. 


### Installation
This repository is based on PyTorch 0.4.1.

### Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/yulequan/UA-MT.git
   cd UA-MT
   ```
2. Put the data in `data/2018LA_Seg_TrainingSet`.
   
3. Train the model:
 
   ```shell
   cd code
   python train_LA_meanteacher_certainty_unlabel.py --gpu 0
   ```

## Citation

If UA-MT is useful for your research, please consider citing:

    @inproceedings{yu2018pu,
         title={Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation},
         author={Yu, Lequan and Wang, Shujun and Li, Xiaomeng and Fu, Chi-Wing and Heng, Pheng-Ann},
         booktitle = {MICCAI},
         year = {2019} }

### Questions

Please contact 'ylqzd2011@gmail.com'

