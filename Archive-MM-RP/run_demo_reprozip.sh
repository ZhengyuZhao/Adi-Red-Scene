bash ./utils/prepare.sh

python ./examples/demo.py

python ./utils/data_clean.py

python ./utils/dis_map_generation.py -batch_size 256 -datasets 'Places' -gpu 1

python ./utils/adaptive_region_selection.py -datasets 'Places' -window_params 3 1

python ./utils/intra_scale_feature_extraction.py -datasets 'Places' -arches 'alexnet' -selection_types 'adi_red' -resolution 'ori_res' -scales 1 2 3

python ./utils/svm_classification.py -datasets 'Places' -arches 'alexnet' -scales 1 2 3 -T2 150 -T3 100 -selection_types 'adi_red' -thresholds 100 150 -resolution 'ori_res' 

python ./utils/plot.py


