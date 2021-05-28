# Adaptive Discriminative Region Discovery for Scene Recognition

This repository contains the Python implementation of "Adi-Red" approach described in our paper:  
**From Volcano to Toyshop: Adaptive Discriminative Region Discovery for Scene Recognition**  
Zhengyu Zhao and Martha Larson, ACMMM 2018. [[Paper]](https://dl.acm.org/citation.cfm?id=3240698) [[Reproducibility Companion Paper@MM'19]](https://dl.acm.org/doi/10.1145/3343031.3351169)
<p align="center">
  <img src="https://github.com/ZhengyuZhao/Adaptive-Discriminative-Region-Discovery/blob/master/figures/diagram_textwidth.jpg" width='600'>
</p>

Adi-Red can derive discriminative information of the scene image directly from a CNN classifier, and achieved state-of-the-art scene recognition performance on [SUN397](https://groups.csail.mit.edu/vision/SUN/) in terms of Top-1 Acc., by adopting a multi-scale patch feature aggregation pipeline with ResNet50-based feature extractor.

## Implementation

### Overview

This code implements:
 1. Generating discriminative map (Dis-Map) for scene images
 2. Adaptively selecting multi-scale discriminative patches
 3. Aggregating CNN features from both local and global scale to obtain the final image representation
 4. Evaluating the approach to scene image recognition on SUN397 and Places
 
### Prerequisites

In order to run the code, you need:  
1. Python3 (tested with Python 3.7.2 on Ubuntu 16.04.6 LTS), required libraries can be installed by running:  
```pip3 install -r requirements.txt```
2. PyTorch deep learning framework (tested with version 1.0.1) and torchvision (tested with version 0.2.2)
3. CUDA driver and cuDNN package if using GPU (tested using Nvidia P100 with CUDA 8.0 and cuDNN 7.1.2)

To install PyTorch and CUDA packages, please refer to their official websites for compatible versions with your system.

Alternatively, the required environment can be automatically set up by using [ReproZip](https://www.reprozip.org/). The .rpz file can be downloaded [here](https://surfdrive.surf.nl/files/index.php/s/GQ14EzWbWFh01Ks/download), and installed following the instructions below:  
```reprounzip docker setup Adi-Red-Scene.rpz Adi-Red-Experiment```  
```reprounzip docker run Adi-Red-Experiment```

### Usage

1. Navigate to the root folder.```cd Archive-MM-RP``` 
2. An example test that covers the key elements of Adi-Red can be run with ```python examples/demo.py```  
3. To replicate the whole experiments, please run the shell script ```bash run.sh``` 
4. Get detailed explanation of the optional parameters of the python scripts ```python [name_of_script].py -h```

**Note**: The datasets (images and labels) are automatically downloaded into ```Archive-MM-RP/datasets```. All the intermediate (e.g., features) outputs are saved in ```Archive-MM-RP/results/intermediate```, and the final results (automatically generated plots and tables) are saved in ```Archive-MM-RP/results/final```.

### Process

The scripts that are executed in ```run.sh``` are described as follows: 

#### Data preparation

Download the datasets and assign the data into required train/test splits:  

```prepare.sh ```  
```python data_clean.py```

#### Dis-Map generation

Generate the discriminative map (Dis-Map) for the scene image:  
```python dis_map_generation.py -batch_size 256 -datasets 'Places' 'SUN397' -gpu 1```

#### Adaptive region selection

Select the discriminative regions based on the Dis-Map, please run:  
```python adaptive_region_selection.py -datasets 'Places' 'SUN397' -window_params 3 1```

#### CNN feature extraction

Extract the intra-scale CNN features from image or image patch, please run:  
```python intra_scale_feature_extraction.py -batch_size_base 32 gpu -1 -datasets 'SUN397' 'Places' -arches 'alexnet' 'resnet18' 'resnet50' -selection_type 'adi_red' -thresholds 100 150 -resolution 'ori_res' -scales 1 2 3 -pretrain_databases 'PL' 'PL' 'IN'```

#### SVM classification

Evaluate the approaches using SVM, please run:  
```python svm_classification.py -datasets 'SUN397' 'Places' -arches 'alexnet' 'resnet18' 'resnet50' -selection_type 'adi_red' -T2 150 -T3 100 -resolution 'ori_res' -scales 1 2 3 -pretrain_databases 'PL' 'PL' 'IN'```

## Results

1. Top-1 accuracy on SUN397: 

	Networks|Baseline|Adi-Red
	:---:|:---:|:---:
	AlexNet|53.87%|61.51%
	ResNet-18|66.99%|70.88%
	ResNet-50|71.14%|73.32%
	
2. Examples of discriminative patches (in finer local scale) discovered by Adi-Red on Places365-Standard validation set. Different levels of discriminative information such as pattern, object and contextual interaction can be captured.
<p align="center">
<img src="https://github.com/ZhengyuZhao/Adaptive-Discriminative-Region-Discovery/blob/master/figures/dis_patch_examples.png" width='800'>
</p>


## Citation

If you use this approach in your research, please cite:

	@inproceedings{zhao2018volcano,
		  title={From Volcano to Toyshop: Adaptive Discriminative Region Discovery for Scene Recognition},
		  author={Zhao, Zhengyu and Larson, Martha},
		  booktitle={2018 ACM Multimedia Conference on Multimedia Conference},
		  pages={1760--1768},
		  year={2018},
		  organization={ACM}
	}
