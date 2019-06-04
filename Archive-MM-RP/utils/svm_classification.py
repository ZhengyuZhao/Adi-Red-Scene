import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn import svm



import argparse

parser = argparse.ArgumentParser(description = "Intra-scale feature extraction")
parser.add_argument("-datasets", "--datasets", nargs='+',help="Specify the dataset used for evaluation", default=['SUN397','Places'])
parser.add_argument("-arches", "--arches", nargs='+',help="Architecture of the CNN feature extractor", default=['alexnet'])
parser.add_argument("-scales", "--scales", nargs='+',help="The total scales(up to 3), in which the features are extracted. ", default=['1','2','3'])
parser.add_argument("-T2", "--T2",help="The threshold used to select the number of discriminative patches in the coaser local scale", default='')
parser.add_argument("-T3", "--T3",help="The threshold used to select the number of discriminative patches in the finer local scale", default='')
parser.add_argument("-resolution", "--resolution", help="specify the mode of input image resolution ('ori_res' or 'low_res')", default="ori_res")
parser.add_argument("-selection_types", "--selection_types", nargs='+',help="The type of method (adi_red, dense or random) used for patch selection ", default=['adi_red'])
parser.add_argument("-pretrain_databases", "--pretrain_databases", nargs='+', help="Specify the pre-training data (Places(PL) or ImageNet(IN)) of the pre-trained CNN feature extractor", default=['PL','PL','IN'])
args = parser.parse_args()

datasets=args.datasets
arches=args.arches
scales=args.scales
T2=args.T2
T3=args.T3

resolution=args.resolution
selection_types=args.selection_types
pretrain_databases=args.pretrain_databases


#L2-normalisation
def feature_post_processing(features_scale):
    features_scale=Normalizer().transform(features_scale)
    return features_scale
#scaling to each element of the features into the range of [0,1]
def scaling(train_fea):
    idx=np.where(np.any(train_fea,axis=0)==False)
    train_fea=np.delete(train_fea, idx, axis=1)
    f_max=np.amax(train_fea, axis=0)
    f_min=np.amin(train_fea, axis=0)
    return f_max,f_min,idx,train_fea

for dataset in datasets:
    result_path = './results/intermediate/'+dataset+'/'
    if dataset=='Places':
        class_num=365
        splits=['']
    if dataset=='SUN397': 
        class_num=397
        splits=['01','02','03','04','05','06','07','08','09','10']
    for arch in arches:
        for selection_type in selection_types:
            acc_sum=0
            for split in splits:
                if scales==['1']:
                      con='1'
                      features_scale_1_train=np.load(result_path+'total_features_'+'1'+'_'+'train'+'_'+arch+'_'+pretrain_databases[0]+split+'.npy') 
                      features_scale_1_test=np.load(result_path+'total_features_'+'1'+'_'+'test'+'_'+arch+'_'+pretrain_databases[0]+split+'.npy')
                      features_scale_1_train=feature_post_processing(features_scale_1_train)
                      features_scale_1_test=feature_post_processing(features_scale_1_test)
                      features_train=features_scale_1_train
                      features_test=features_scale_1_test
                      del features_scale_1_test,features_scale_1_train
                if scales==['1','2']:
                      con='1_2'
                      features_scale_1_train=np.load(result_path+'total_features_'+'1'+'_'+'train'+'_'+arch+'_'+pretrain_databases[0]+split+'.npy') 
                      features_scale_1_test=np.load(result_path+'total_features_'+'1'+'_'+'test'+'_'+arch+'_'+pretrain_databases[0]+split+'.npy')
                      features_scale_2_train=np.load(result_path+'total_features_'+'2'+'_'+'train'+'_'+arch+'_'+selection_type+'_'+T2+'_'+pretrain_databases[1]+'_'+resolution+split+'.npy') 
                      features_scale_2_test=np.load(result_path+'total_features_'+'2'+'_'+'test'+'_'+arch+'_'+selection_type+'_'+T2+'_'+pretrain_databases[1]+'_'+resolution+split+'.npy')
                      features_scale_1_train=feature_post_processing(features_scale_1_train)
                      features_scale_1_test=feature_post_processing(features_scale_1_test)
                      features_scale_2_train=feature_post_processing(features_scale_2_train)
                      features_scale_2_test=feature_post_processing(features_scale_2_test)
                      features_train=np.concatenate((features_scale_1_train,features_scale_2_train),axis=1)
                      features_test=np.concatenate((features_scale_1_test,features_scale_2_test),axis=1)
                      del features_scale_1_test,features_scale_2_test,features_scale_1_train,features_scale_2_train
                
                if scales==['1','3']:
                      con='1_3'
                      features_scale_1_train=np.load(result_path+'total_features_'+'1'+'_'+'train'+'_'+arch+'_'+pretrain_databases[0]+split+'.npy') 
                      features_scale_1_test=np.load(result_path+'total_features_'+'1'+'_'+'test'+'_'+arch+'_'+pretrain_databases[0]+split+'.npy')
                      features_scale_3_train=np.load(result_path+'total_features_'+'3'+'_'+'train'+'_'+arch+'_'+selection_type+'_'+T3+'_'+pretrain_databases[2]+'_'+resolution+split+'.npy') 
                      features_scale_3_test=np.load(result_path+'total_features_'+'3'+'_'+'test'+'_'+arch+'_'+selection_type+'_'+T3+'_'+pretrain_databases[2]+'_'+resolution+split+'.npy')
                      features_scale_1_train=feature_post_processing(features_scale_1_train)
                      features_scale_1_test=feature_post_processing(features_scale_1_test)
                      features_scale_3_train=feature_post_processing(features_scale_3_train)
                      features_scale_3_test=feature_post_processing(features_scale_3_test)
                      features_train=np.concatenate((features_scale_1_train,features_scale_3_train),axis=1)
                      features_test=np.concatenate((features_scale_1_test,features_scale_3_test),axis=1)
                      del features_scale_1_test,features_scale_3_test,features_scale_1_train,features_scale_3_train
                
                if scales==['1','2','3']:
                      con='1_2_3'
                      features_scale_1_train=np.load(result_path+'total_features_'+'1'+'_'+'train'+'_'+arch+'_'+pretrain_databases[0]+split+'.npy') 
                      features_scale_1_test=np.load(result_path+'total_features_'+'1'+'_'+'test'+'_'+arch+'_'+pretrain_databases[0]+split+'.npy')
                      features_scale_2_train=np.load(result_path+'total_features_'+'2'+'_'+'train'+'_'+arch+'_'+selection_type+'_'+T2+'_'+pretrain_databases[1]+'_'+resolution+split+'.npy') 
                      features_scale_2_test=np.load(result_path+'total_features_'+'2'+'_'+'test'+'_'+arch+'_'+selection_type+'_'+T2+'_'+pretrain_databases[1]+'_'+resolution+split+'.npy')
                      features_scale_3_train=np.load(result_path+'total_features_'+'3'+'_'+'train'+'_'+arch+'_'+selection_type+'_'+T3+'_'+pretrain_databases[2]+'_'+resolution+split+'.npy') 
                      features_scale_3_test=np.load(result_path+'total_features_'+'3'+'_'+'test'+'_'+arch+'_'+selection_type+'_'+T3+'_'+pretrain_databases[2]+'_'+resolution+split+'.npy')          
                      features_scale_1_train=feature_post_processing(features_scale_1_train)
                      features_scale_1_test=feature_post_processing(features_scale_1_test)
                      features_scale_2_train=feature_post_processing(features_scale_2_train)
                      features_scale_2_test=feature_post_processing(features_scale_2_test)
                      features_scale_3_train=feature_post_processing(features_scale_3_train)
                      features_scale_3_test=feature_post_processing(features_scale_3_test)
                      
                      features_train=np.concatenate((features_scale_1_train,features_scale_2_train),axis=1)
                      del features_scale_1_train,features_scale_2_train
                      features_train=np.concatenate((features_train,features_scale_3_train),axis=1)
                      del features_scale_3_train
                      features_test=np.concatenate((features_scale_1_test,features_scale_2_test),axis=1)
                      del features_scale_1_test,features_scale_2_test
                      features_test=np.concatenate((features_test,features_scale_3_test),axis=1)          
                      del features_scale_3_test
                      
                f_max,f_min,idx,features_train=scaling(features_train)   
                features_test=np.delete(features_test, idx, axis=1)
                features_train=(features_train-f_min)/(f_max-f_min)
                features_test=(features_test-f_min)/(f_max-f_min)
                labels=[]          
                for i in range(50*class_num):
                    labels.append(i//50)
                model = svm.LinearSVC(C=0.02,dual=False,tol=0.01,fit_intercept=False)
                model.fit(features_train, labels) 
                acc=model.score(features_test,labels)
                print(acc)
                acc_sum=acc+acc_sum
            acc_avg=acc_sum/len(splits)*100
            print(acc_avg)
            #save the accuracy on Places or the average accuracy over 10 spilts on SUN397
            if scales==['1']:
                np.save(result_path+'acc_'+con+'_'+arch+'_'+pretrain_databases[0]+'.npy',acc_avg)    
            if scales==['1','2']:
                np.save(result_path+'acc_'+con+'_'+arch+'_'+selection_type+'_'+T2+'_'+pretrain_databases[0]+'_'+pretrain_databases[1]+'_'+resolution+'.npy',acc_avg)    
            if scales==['1','3']:  
                np.save(result_path+'acc_'+con+'_'+arch+'_'+selection_type+'_'+T3+'_'+pretrain_databases[0]+'_'+pretrain_databases[2]+'_'+resolution+'.npy',acc_avg)    
            if scales==['1','2','3']:    
                np.save(result_path+'acc_'+con+'_'+arch+'_'+selection_type+'_'+T2+'_'+T3+'_'+pretrain_databases[0]+'_'+pretrain_databases[1]+'_'+pretrain_databases[2]+'_'+resolution+'.npy',acc_avg)    
