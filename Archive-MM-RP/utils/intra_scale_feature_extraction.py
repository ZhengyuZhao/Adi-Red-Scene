import torch
import torchvision.models as models
from torchvision import transforms as trn
import torch.nn as nn 
import numpy as np
from PIL import Image
import os
from tqdm import tqdm


import argparse

parser = argparse.ArgumentParser(description = "Intra-scale feature extraction")
parser.add_argument("-batch_size_base", "--batch_size_base", type=int, help="Number of images processed at one time", default=32)
parser.add_argument("-datasets", "--datasets", nargs='+',help="Specify the dataset used for evaluation", default=['SUN397','Places'])
parser.add_argument("-gpu", "--gpu", type=int, help="1 for gpu and -1 for cpu", default=1)
parser.add_argument("-arches", "--arches", nargs='+',help="Architecture of the CNN feature extractor", default=['alexnet'])
parser.add_argument("-scales", "--scales", nargs='+',help="The total scales(up to 3), in which the features are extracted. ", default=['1','2','3'])
parser.add_argument("-thresholds", "--thresholds", nargs='+',help="The threshold used to select the number of discriminative patches", default=['100','150'])
parser.add_argument("-resolution", "--resolution", help="specify the mode of input image resolution ('ori_res' or 'low_res' ", default="ori_res")
parser.add_argument("-selection_types", "--selection_types", nargs='+',help="The type of method (adi_red, dense or random) used for patch selection ", default=['adi_red'])
parser.add_argument("-pretrain_databases", "--pretrain_databases",nargs='+', help="Specify the pre-training data (Places(PL) or ImageNet(IN)) of the pre-trained CNN feature extractor", default=['PL','PL','IN'])
args = parser.parse_args()
batch_size_base=args.batch_size_base
datasets=args.datasets
arches=args.arches
scales=args.scales
thresholds=args.thresholds
resolution=args.resolution
selection_types=args.selection_types
pretrain_databases=args.pretrain_databases



if  args.gpu==1:   
    device  = torch.device("cuda:0")
if  args.gpu==-1:
    device  = torch.device("cpu")
def  returnTF(scale,resolution):
# load the image transformer
    if scale=='1':
       scale_image_size=224
    if scale=='2':
       scale_image_size=448
    if scale=='3':
       scale_image_size=896
    if resolution=='ori_res' :  
        tf = trn.Compose([
            trn.Resize((scale_image_size,scale_image_size)),
            trn.ToTensor (),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    if resolution=='low_res' : 
        if scale=='1':
            tf = trn.Compose([
                trn.Resize((224,224)),
                trn.ToTensor (),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        if scale=='2' or scale=='3':
            tf = trn.Compose([
                trn.Resize((224,224)),
                trn.Resize((scale_image_size,scale_image_size)),
                trn.ToTensor (),
                trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    return tf

def load_model(arch,pretrain_database):
    if pretrain_database=='PL':
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
             os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)        
        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
    elif pretrain_database=='IN':
        if arch=='resnet50':
          model=models.resnet50(pretrained=True)
        elif arch=='resnet18':
          model=models.resnet18(pretrained=True)
        elif arch=='alexnet':
          model=models.alexnet(pretrained=True)

    if arch=='resnet50' or arch=='resnet18':
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
    if arch=='alexnet':
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
        feature_extractor=model
    feature_extractor.eval()

    return feature_extractor



modes=['train','test']
for dataset in datasets:
    image_path='./datasets/images/'+dataset+'/'
    label_path='./datasets/labels/'+dataset+'/'
    result_path = './results/intermediate/'+dataset+'/'
    if dataset=='SUN397':
        splits=['01','02','03','04','05','06','07','08','09','10']#10 fixed train/test splits for evaluation on SUN397 
        with open(label_path+'image_list_all.txt', 'r') as f:
            image_list = f.readlines()
    for arch in arches:
        for selection_type in selection_types:            
            if selection_type=='adi_red':      
                for scale in scales:
                    if arch == 'alexnet':
                        fea_dim=4096
                        if scale=='1':
                            batch_size_ori = 16*batch_size_base
                        if scale=='2' or scale=='3':
                            batch_size_ori = 2*batch_size_base
                    if arch == 'resnet18':
                        fea_dim=512
                        if scale=='1':
                            batch_size_ori = 4*batch_size_base
                        if scale=='2' or scale=='3':
                            batch_size_ori = batch_size_base
                    if arch == 'resnet50':
                        fea_dim=2048
                        if scale=='1':
                            batch_size_ori = 4*batch_size_base
                        if scale=='2' or scale=='3':
                            batch_size_ori = batch_size_base 
                    feature_extractor = load_model(arch,pretrain_databases[int(scale)-1]) 
                    feature_extractor=feature_extractor.to(device)                             
                    for mode in modes:
                        if dataset=='Places':
                           with open(label_path+'image_list_'+mode+'.txt', 'r') as f:
                               image_list = f.readlines()
                        num_images = len(image_list) 
                        if scale=='2' or scale =='3':
                            local_maxima=np.load(result_path+'local_max_'+scale+'_'+mode+'.npy')
                            total_features=[]
                            for tt,T in enumerate(thresholds):
                              total_features.append(np.zeros((num_images, fea_dim)))
                        if scale=='1':                       
                            total_features = np.zeros((num_images, fea_dim))
                        num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size_ori)))
                        for k in tqdm(range(0, num_batches),desc="Fea_ext"):
                            data_inp = image_list[k*batch_size_ori:min((k+1)*batch_size_ori,num_images)]
                            if scale=='2' or scale =='3':
                                cord_batch=local_maxima[k*batch_size_ori:min((k+1)*batch_size_ori,num_images)] 
                            batch_size=len(data_inp)                    
                            if scale=='2' or scale =='3':
                                patch_numbers=np.zeros(batch_size,dtype=int)
                                for t in range(0,batch_size):
                                    patch_numbers[t]=len(cord_batch[t][0])
                                img_patches_batch = torch.zeros(int(sum(patch_numbers)),3,224,224)
                                sum_list=0
                            if scale=='1':  
                                input_images_batch = torch.zeros(batch_size,3,224,224)
                            for i,image_name in enumerate(data_inp):  
                                if dataset=='SUN397':
                                    img = Image.open(image_path+image_name[:-1])   
                                if dataset=='Places':
                                    img = Image.open(image_path+image_name.split(' ')[0])   
                                img_RGB=img.convert('RGB')                       
                                input_image = returnTF(scale,resolution)(img_RGB).unsqueeze(0)
                                if scale=='2' or scale =='3':
                                    top=cord_batch[i][0]
                                    left=cord_batch[i][1]
                                    for j in range(sum_list,sum_list+patch_numbers[i]):
                                       img_patches_batch[j]=input_image[:,:,top[j-sum_list]:top[j-sum_list]+224,left[j-sum_list]:left[j-sum_list]+224]
                                    sum_list=sum_list+patch_numbers[i]
                                if scale=='1':  
                                    input_images_batch[i]=input_image
                            if scale=='2' or scale =='3':       
                                features = feature_extractor(img_patches_batch.to(device))
                                features=features.view(int(sum_list),-1).cpu().detach().numpy()
                                cnt_list=0
                                for t in range(0,batch_size):
                                    loc_m=np.array(cord_batch[t][2])
                                    for tt,T in enumerate(thresholds):                                         
                                        pos=np.where(loc_m>int(T))
                                        if len(pos[0])==0:
                                            total_features[tt][k*batch_size+t]=np.zeros((1,fea_dim))
                                        else:    
                                            total_features[tt][k*batch_size+t]=np.amax(features[cnt_list:cnt_list+patch_numbers[t]][pos], axis=0)
                                    cnt_list=cnt_list+patch_numbers[t]
                            if scale=='1':
                                features = feature_extractor(input_images_batch.to(device))
                                features=features.view(batch_size,-1).cpu().detach().numpy()
                                total_features[k*batch_size:min((k+1)*batch_size,num_images)]=features
                        if scale=='1':                        
                            np.save(result_path+'total_features_'+scale+'_'+mode+'_'+arch+'_'+pretrain_databases[int(scale)-1]+'.npy', total_features)
                            if dataset=='SUN397':
                                for split in splits:
                                    split_index=np.load(label_path+'index_'+mode+'_'+split+'.npy')                          
                                    np.save(result_path+'total_features_'+scale+'_'+mode+'_'+arch+'_'+pretrain_databases[int(scale)-1]+split+'.npy',total_features[split_index])
                        if scale=='2' or scale=='3':
                            for tt,T in enumerate(thresholds):
                                np.save(result_path+'total_features_'+scale+'_'+mode+'_'+arch+'_'+selection_type+'_'+T+'_'+pretrain_databases[int(scale)-1]+'_'+resolution+'.npy',total_features[tt])
                                if dataset=='SUN397':
                                    for split in splits:
                                        split_index=np.load(label_path+'index_'+mode+'_'+split+'.npy')                          
                                        np.save(result_path+'total_features_'+scale+'_'+mode+'_'+arch+'_'+selection_type+'_'+T+'_'+pretrain_databases[int(scale)-1]+'_'+resolution+split+'.npy',total_features[tt][split_index])
            if selection_type=='random' or selection_type=='dense':            
                for scale in scales:
                    if scale=='1':
                      continue
                    if scale=='2':
                      scale_image_size=448
                    if scale=='3':
                      scale_image_size=896
                    if arch == 'alexnet':
                        fea_dim=4096
                    if arch == 'resnet18':
                        fea_dim=512
                    if arch == 'resnet50':
                        fea_dim=2048
                    feature_extractor = load_model(arch,pretrain_databases[int(scale)-1]) 
                    feature_extractor=feature_extractor.to(device)                             
                    for mode in modes:
                        if dataset=='Places':
                           with open(label_path+'image_list_'+mode+'.txt', 'r') as f:
                               image_list = f.readlines()
                        num_images = len(image_list)       
                        total_features = np.zeros((num_images, fea_dim))
                        if selection_type=='random':                            
                            patch_numbers=5
                            batch_size_ori = 8*batch_size_base
                        if selection_type=='dense':                      
                            if scale=='3':
                                patch_numbers=50
                                batch_size_ori = batch_size_base
                            if scale=='2':
                                patch_numbers=10
                                batch_size_ori = 4*batch_size_base
                        num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size_ori)))    
                        for k in tqdm(range(0, num_batches),desc="Fea_ext"):
                            data_inp = image_list[k*batch_size_ori:min((k+1)*batch_size_ori,num_images)]
                            batch_size=len(data_inp)                    
                            patch_numbers_batch=patch_numbers*batch_size
                            img_patches_batch = torch.zeros(patch_numbers_batch,3,224,224)
                            top=np.random.randint(0, scale_image_size - 223,size=patch_numbers)
                            left=np.random.randint(0, scale_image_size - 223,size=patch_numbers)
                            for i,image_name in enumerate(data_inp):                    
                                if dataset=='SUN397':
                                    img = Image.open(image_path+image_name[:-1])   
                                if dataset=='Places':
                                    img = Image.open(image_path+image_name.split(' ')[0])       
                                img_RGB=img.convert('RGB')                       
                                input_image = returnTF(scale,resolution)(img_RGB).unsqueeze(0)
                                for j in range(0,patch_numbers):
                                    img_patches_batch[i*patch_numbers+j]=input_image[:,:,top[j]:top[j]+224,left[j]:left[j]+224]                                  
                            features = feature_extractor(img_patches_batch.to(device))
                            features=features.view(int(patch_numbers_batch),-1).cpu().detach().numpy()
                            for t in range(0,batch_size):    
                                  total_features[k*batch_size+t]=np.amax(features[t*patch_numbers:(t+1)*patch_numbers], axis=0)                      
                        np.save(result_path+'total_features_'+scale+'_'+mode+'_'+arch+'_'+selection_type+'_'+'_'+pretrain_databases[int(scale)-1]+'_'+resolution+'.npy', total_features)   
                        if dataset=='SUN397':
                            for split in splits:
                                split_index=np.load(label_path+'index_'+mode+'_'+split+'.npy')
                                np.save(result_path+'total_features_'+scale+'_'+mode+'_'+arch+'_'+selection_type+'_'+'_'+pretrain_databases[int(scale)-1]+'_'+resolution+split+'.npy',total_features[split_index])                                
