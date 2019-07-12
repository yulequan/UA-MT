# Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation
by [Lequan Yu](http://yulequan.github.io), [Shujun Wang](https://emmaw8.github.io/), [Xiaomeng Li](https://xmengli999.github.io/), [Chi-Wing Fu](http://www.cse.cuhk.edu.hk/~cwfu/), [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/). 

### Introduction

This repository is for our MICCAI 2019 paper '[Uncertainty-aware Self-ensembling Model for Semi-supervised 3D Left Atrium Segmentation](XX)'. 

Coming Soon. 

<!---

### Installation
This repository is based on Tensorflow and the TF operators from PointNet++. Therefore, you need to install tensorflow and compile the TF operators. 

For installing tensorflow, please follow the official instructions in [here](https://www.tensorflow.org/install/install_linux). The code is tested under TF1.3 (higher version should also work) and Python 2.7 on Ubuntu 16.04.

For compiling TF operators, please check `tf_xxx_compile.sh` under each op subfolder in `code/tf_ops` folder. Note that you need to update `nvcc`, `python` and `tensoflow include library` if necessary. You also need to remove `-D_GLIBCXX_USE_CXX11_ABI=0` flag in g++ command in order to compile correctly if necessary.

To compile the operators in TF version >=1.4, you need to modify the compile scripts slightly.

First, find Tensorflow include and library paths.

        TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
        TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
        
Then, add flags of `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` to the `g++` commands. You can refer [tf_cd_compile.sh](https://github.com/yulequan/PU-Net/blob/master/code/tf_ops/CD/tf_cd_compile.sh).

### Note
When running the code, if you have `undefined symbol: _ZTIN10tensorflow8OpKernelE` error, you need to compile the TF operators. If you have already added the `-I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework` but still have ` cannot find -ltensorflow_framework` error. Please use 'locate tensorflow_framework
' to locate the tensorflow_framework library and make sure this path is in `$TF_LIB`.

### Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/yulequan/PU-Net.git
   cd PU-Net
   ```
2. Compile the TF operators
   Follow the above information to compile the TF operators. 
   
3. Train the model:
  First, you need to download the training patches in HDF5 format from [GoogleDrive](https://drive.google.com/file/d/1te8d1y2BTFBL_3CB1jpqbOFzkkjvtKsE/view?usp=sharing) and put it in folder `h5_data`.
  Then run:
   ```shell
   cd code
   python main.py --phase train
   ```

4. Evaluate the model:
    First, you need to download the pretrained model from [GoogleDrive](https://drive.google.com/file/d/1c1oYNwIzKxCOF_6bqm3HmwYcCZv1230Z/view?usp=sharing), extract it and put it in folder 'model'.
    Then run:
   ```shell
   cd code
   python main.py --phase test --log_dir ../model/generator2_new6
   ```
   You will see the input and output results in the folder `../model/generator2_new6/result`.
   
5. The training and testing mesh files can be downloaded from [GoogleDrive](https://drive.google.com/file/d/1lp2HWU78tx_-tz8cxE8PSxb-M7ZQCMOq/view?usp=sharing).

Note: In this version, we treat the whole input point cloud as a single input. If the number of points in your input point cloud is big, you had better divide the input point cloud into patches and treat each patch as a single input. (see our following work [EC-Net](https://github.com/yulequan/EC-Net))

### Evaluation code
We provide the code to calculate the metric NUC in the evaluation code folder. In order to use it, you need to install the CGAL library. Please refer [this link](https://www.cgal.org/download/linux.html) to install this library.
Then:
   ```shell
   cd evaluation_code
   cmake .
   make
   ./evaluation nicolo.off nicolo.xyz
```
The second argument is the mesh, and the third one is the predicted points. 

After running this program, the distances of each predicted point to the surface are written in `nicolo_point2mesh_distance.xyz`, and the density of each disk (n_i/N) are written in `nicolo_density.xyz`.

## Citation

If PU-Net is useful for your research, please consider citing:

    @inproceedings{yu2018pu,
         title={PU-Net: Point Cloud Upsampling Network},
         author={Yu, Lequan and Li, Xianzhi and Fu, Chi-Wing and Cohen-Or, Daniel and Heng, Pheng-Ann},
         booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
         year = {2018}
   }

### Questions

Please contact 'lqyu@cse.cuhk.edu.hk'

-->
