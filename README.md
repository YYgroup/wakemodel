# CNN wake model

This data-driven model is based on the convolutional neural network (CNN) for force estimation in moving boundary problems by regarding the velocity field on a series of cross sections as images. It has three CNN architectures for integrating physical information or attention mechanism. The model performances indicate that the optimized CNN can identify important flow regions and learn empirical physical laws.


## Environment
- python3.7
- pytorch1.0+cu80


## Usage

### 1. dataset
Prepare the input data from the DNS with different flow parameters.

Create a root directory (e.g. data/stationary_plate/npy_files), and case directory (e.g. re300_aoa5_ar2), then prepare the data along the streamwise position in the wake. For the velocity data, its shape is (H, W, 3) with three components.

The data structure looks like
```
│ data
    | stationary_plate
        | npy_files
            | re300_aoa5_ar2
                | npy_velocity_x_ 1.00_index_01000.npy
                | npy_velocity_x_ 1.02_index_01000.npy
                | npy_velocity_x_ 1.04_index_01000.npy
                | ...
            | re300_aoa5_ar2
            | re300_aoa10_ar2
            | ...
```


### 2. train and test the model
The baseline CNN model is adapted from the ResNet.You can run the provided scripts with different model architectures to train and test the model.

```bash
$ sh st_data1_resnet9.sh  # the baseline
$ sh st_data1_resnet9_aoa.sh  # baseline + alpha
$ sh st_data1_resnet9_aoa_spatt.sh # baseline + alpha + attention module
```

## Acknowledgement
This repository is based on [pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100).