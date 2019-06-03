bash ./utils/prepare.sh

python ./utils/data_clean.py

python ./utils/dis_map_generation.py -batch_size 256 -datasets 'Places' 'SUN397' -gpu 1

python ./utils/adaptive_region_selection.py -datasets 'Places' 'SUN397' -window_params 3 1

#results for Table 1
python ./utils/intra_scale_feature_extraction.py -batch_size_base 32 -datasets 'SUN397' -arches 'alexnet' -selection_types 'adi_red' 'dense' 'random' -scales 1 2 3 -thresholds 100 150 -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'

python ./utils/svm_classification.py -datasets 'SUN397' -arches 'alexnet' -scales 1 2 -T2 150 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'
python ./utils/svm_classification.py -datasets 'SUN397' -arches 'alexnet' -scales 1 2 -selection_types 'dense' 'random' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'

python ./utils/svm_classification.py -datasets 'SUN397' -arches 'alexnet' -scales 1 3 -T3 100 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'
python ./utils/svm_classification.py -datasets 'SUN397' -arches 'alexnet' -scales 1 3  -selection_types 'dense' 'random' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'

python ./utils/svm_classification.py -datasets 'SUN397' -arches 'alexnet' -scales 1 2 3 -T2 150 -T3 100 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN' 
python ./utils/svm_classification.py -datasets 'SUN397' -arches 'alexnet' -scales 1 2 3 -selection_types 'dense' 'random' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN' 

#results for Table 2
python ./utils/intra_scale_feature_extraction.py -batch_size_base 16 -datasets 'SUN397' -arches 'resnet18' 'resnet50' -selection_types 'adi_red' -scales 1 2 3 -thresholds 100 150 -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'

python ./utils/svm_classification.py -datasets 'SUN397' -arches 'alexnet' 'resnet18' 'resnet50' -scales 1 -pretrain_databases

python ./utils/svm_classification.py -datasets 'SUN397' -arches 'resnet18' 'resnet50' -scales 1 2 3 -T2 150 -T3 100 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'

#results for Figure 3

python ./utils/intra_scale_feature_extraction.py -datasets 'Places' -arches 'alexnet' -selection_types 'adi_red' -scales 1 2 3 -thresholds 0 50 100 150 200 225 -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'

python ./utils/svm_classification.py -datasets 'Places' -arches 'alexnet' -scales 1 2 3 -T2 0 -T3 0 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'
python ./utils/svm_classification.py -datasets 'Places' -arches 'alexnet' -scales 1 2 3 -T2 50 -T3 50 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'
python ./utils/svm_classification.py -datasets 'Places' -arches 'alexnet' -scales 1 2 3 -T2 100 -T3 100 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'
python ./utils/svm_classification.py -datasets 'Places' -arches 'alexnet' -scales 1 2 3 -T2 150 -T3 150 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'
python ./utils/svm_classification.py -datasets 'Places' -arches 'alexnet' -scales 1 2 3 -T2 200 -T3 200 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'
python ./utils/svm_classification.py -datasets 'Places' -arches 'alexnet' -scales 1 2 3 -T2 225 -T3 225 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'

#results for Table 4
python ./utils/intra_scale_feature_extraction.py -datasets 'Places' -arches 'alexnet' -selection_types 'adi_red' -scales 2 3 -thresholds 100 150 -resolution 'ori_res' -pretrain_databases 'PL' 'IN' 'PL'

python ./utils/svm_classification.py -datasets 'Places' -arches 'alexnet' -scales 1 -pretrain_databases 'PL' 'PL' 'IN'
python ./utils/svm_classification.py -datasets 'Places' -arches 'alexnet' -scales 1 3 -T3 100 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'
python ./utils/svm_classification.py -datasets 'Places' -arches 'alexnet' -scales 1 2 -T2 150 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'
python ./utils/svm_classification.py -datasets 'Places' -arches 'alexnet' -scales 1 2 3 -T2 150 -T3 100 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'IN' 'IN'
python ./utils/svm_classification.py -datasets 'Places' -arches 'alexnet' -scales 1 2 3 -T2 150 -T3 100 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'PL'
python ./utils/svm_classification.py -datasets 'Places' -arches 'alexnet' -scales 1 2 3 -T2 150 -T3 100 -selection_types 'adi_red' -resolution 'ori_res' -pretrain_databases 'PL' 'PL' 'IN'


#plot four tables (original Table 1, Table 2 and Table 4, and table for the resolution experiment) and two graphs (Figure 3 and Figure 4) 

python ./utils/plot.py
