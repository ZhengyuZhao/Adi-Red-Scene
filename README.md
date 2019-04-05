# Adaptive Discriminative Region Discovery for Scene Recognition

This repository contains the Python implementation of "Adi-Red" approach described in our paper:  
**From Volcano to Toyshop: Adaptive Discriminative Region Discovery for Scene Recognition**  
Zhengyu Zhao and Martha Larson, ACMMM 2018. [[Paper]](https://dl.acm.org/citation.cfm?id=3240698)
<p align="center">
  <img src="https://github.com/ZhengyuZhao/Adaptive-Discriminative-Region-Discovery/blob/master/figures/diagram_textwidth.jpg" width='600'>
</p>

Adi-Red can derive discriminative information of the scene image directly from a CNN classifier, and achieved state-of-the-art scene recognition performance on [SUN397](https://groups.csail.mit.edu/vision/SUN/) in terms of Top-1 Acc., by adopting a multi-scale patch feature aggregation pipeline with ResNet50-based feature extractor.

## Implementation

### Overview

This code implements:
 1. Generating discriminative map (Dis-Map) for scene images
 2. Aadptively selecting multi-scale discriminative patches
 3. Aggregating CNN features from both local and global scale to obtain the final image representation
 4. Evaluating the scene image recognition on SUN397 and Places
 
### Prerequisites

In order to run the code, you need:  
1. Python3 (tested with Python 3.7.2 on Ubuntu 16.04.6 LTS)
2. PyTorch deep learning framework (tested with version 1.0.1)
3. All the rest (data + networks) will be automatically downloaded with our scripts

### Usage

1. Navigate to the root folder of the code.```cd Archive-MM-RP```  
2. Run the bash file ```bash prepare.sh``` to construct the file structure and download the datasets.  
3. Get detailed explanation of the optional parameters of the python scripts```python [name_of_script].py -h```

### Splits assignment

To assign the data into required train/test splits, please run:  
```python data_clean.py```

### Dis-Map generation

To generate the discriminative map (Dis-Map) for the scene image, please run:  
```python dis_map_generation.py -batch_size 256 -datasets 'Places' 'SUN397' -gpu 1```

### Adaptive region selection

To select the discriminative regions based on the Dis-Map, please run:  
```python adaptive_region_selection.py -datasets 'Places' 'SUN397' -window_params ['3','1']```

### CNN feature extraction

To extract the intra-scale CNN features from image or image patch, please run:  
```python intra_scale_feature_extraction.py -batch_size_base 32 gpu -1 -datasets 'SUN397' 'Places' -arches 'alexnet' 'resnet18' 'resnet50' -selection_type 'adi_red' -thresholds 100 150 -resolution 'ori_res' -scales 1 2 3 -pretrain_databases ['PL','PL','IN']```

### SVM classification

To evaluate the approach using SVM, please run:  
```python svm_classification.py -datasets 'SUN397' 'Places' -arches 'alexnet' 'resnet18' 'resnet50' -selection_type 'adi_red' -T2 150 -T3 100 -resolution 'ori_res' -scales 1 2 3 -pretrain_databases ['PL','PL','IN']```

**Note**: The datasets (images and labels) are automatically downloaded into ```Archive-MM-RP/datasets``` and all the output results (Dis-Maps, patch locations, features and accuracy numbers) are saved in ```Archive-MM-RP/results```


## Results

1. Top-1 accuracy on SUN397:  
	Networks|Baseline|Adi-Red
	:---:|:---:|:---:
	AlexNet|%|%
	ResNet-18|%|%
	ResNet-50|%|%
	
2. Examples of discriminative patches (in finer local scale) discovered by Adi-Red on Places365-Standard validation set. Different levels of discriminative information such as pattern, object and contextual interaction can be captured.
<p align="center">
<img src="https://github.com/ZhengyuZhao/Adaptive-Discriminative-Region-Discovery/blob/master/figures/dis_patch_examples.png" width='800'>
</p>


## Citation

If you use this approach in your research, please cite:

	@article{Zhao2018,
		author = {Zhengyu Zhao and Martha Larson},
		title = {From Volcano to Toyshop: Adaptive Discriminative Region Discovery for Scene Recognition},
		Booktitle={ACM International Conference on Multimedia},
		Year={2018}
	}
