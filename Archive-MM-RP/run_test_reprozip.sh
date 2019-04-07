bash ./utils/prepare.sh

python ./utils/data_clean.py

python ./utils/dis_map_generation.py -batch_size 256 -datasets 'SUN397' 'Places' -gpu 1

python ./utils/adaptive_region_selection.py -datasets 'SUN397' 'Places' -window_params 3 1

python ./utils/intra_scale_feature_extraction.py -datasets 'SUN397' -arches 'resnet18' -selection_types 'adi_red' -thresholds 100 150 -resolution 'ori_res' -scales 1 2 3

python ./utils/svm_classification.py -datasets 'SUN397' -arches 'resnet18' -scales 1 2 3 -T2 150 -T3 100 -selection_types 'adi_red' -thresholds 100 150 -resolution 'ori_res' 

python ./utils/plot_test.py


