import torch
from torchvision import transforms as trn
from torch.nn import functional as F
import torch.nn as nn
#import the
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

import argparse
import warnings
warnings.filterwarnings('ignore', 'Possibly corrupt EXIF data*')
parser = argparse.ArgumentParser(description = "Generating the discriminative map per image")
parser.add_argument("-batch_size", "--batch_size_ori", type=int, help="Number of images processed at one time", default=256)
parser.add_argument("-datasets", "--datasets", nargs='+',help="Specify the dataset", default=['Places','SUN397'])
parser.add_argument("-gpu", "--gpu", type=int, help="1 for gpu and -1 for cpu", default=1)
args = parser.parse_args()




if  args.gpu==1:   
    device  = torch.device("cuda:0")
if  args.gpu==-1:
    device  = torch.device("cpu")

model_file = 'wideresnet18_places365.pth.tar'
batch_size_ori=args.batch_size_ori
datasets=args.datasets

def returnCAM(feature_conv, weight_softmax, class_idx):
    nc, h, w = feature_conv.shape
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)    
    cam = cam - np.min (cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    return cam_img

def  returnTF ():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
        trn.ToTensor (),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf

#load the DisNet used for generating Dis-Map
def load_model(model_file):
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
    os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py'+' -P ./utils')
    import wideresnet
    model = wideresnet.resnet18(num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval() 
    for param in model.parameters():
        param.requires_grad = False
    params = list(model.parameters())
    weight_softmax = params[-2].clone().numpy()
    weight_softmax[weight_softmax<0] = 0    

    conv_model=nn.Sequential(*list(model.children())[:-2])
    linear_layer=model.fc
    conv_model.eval()
    linear_layer.eval()
    return linear_layer,conv_model,weight_softmax

#
tf=returnTF ()
linear_layer,conv_model,weight_softmax = load_model(model_file)
linear_layer=linear_layer.to(device)
conv_model=conv_model.to(device)

for dataset in datasets:
    if dataset=='SUN397':
        image_path='./datasets/images/SUN397/'
        label_path='./datasets/labels/SUN397/'
        result_path = './results/intermediate/SUN397/'
        with open(label_path+'ClassName.txt', 'r') as f:
                  class_labels = f.readlines() 
        with open(label_path+'image_list_all.txt', 'r') as f:
                  image_list_all = f.readlines()
        for a in range(0,len(class_labels)) : 
           class_labels[a]=  class_labels[a][:-1]   
        CAM_map_all_train=[]
        CAM_map_all_test=[]
        
        num_images = len(image_list_all)
        num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size_ori)))
        Class_overlap=np.load(label_path+'Class_overlap.npy')
        for k in tqdm(range(0, num_batches)):
                data_inp = image_list_all[k*batch_size_ori:min((k+1)*batch_size_ori,num_images)]
                batch_size=len(data_inp)
                input_img = torch.zeros(batch_size,3,224,224)
                GT_labels=np.zeros(batch_size,dtype=int)
                for i,image_name in enumerate(data_inp):
            
                    img = Image.open(image_path+image_name[:-1])       
                    img_RGB=img.convert('RGB')
                    input_img[i] = tf(img_RGB).unsqueeze(0)
                    GT_labels[i] = Class_overlap[class_labels.index(image_name[:-26])]     
                    
                conv_maps = conv_model(input_img.to(device))
                conv_maps_np=conv_maps.cpu().detach().numpy()
                logits=linear_layer(nn.AvgPool2d(kernel_size=14)(conv_maps).view(-1, 512))
                h_xs=F.softmax(logits,dim=1).cpu().detach()
                for t in range(0,batch_size):
                     probs, idx = (h_xs[t]).sort(0, True)
                     CAM_map_test= returnCAM(conv_maps_np[t], weight_softmax,idx[0])
                     if GT_labels[t]==-1:
                         CAM_map_train= CAM_map_test
                     else:
                         CAM_map_train= returnCAM(conv_maps_np[t], weight_softmax,GT_labels[t])
                     CAM_map_all_train.append(CAM_map_train)
                     CAM_map_all_test.append(CAM_map_test)
        #         print(k)
        np.save(result_path+'CAM_map_all_train.npy',CAM_map_all_train)             
        np.save(result_path+'CAM_map_all_test.npy',CAM_map_all_test)
    if dataset=='Places':
        image_path='./datasets/images/Places/'
        label_path='./datasets/labels/Places/'
        result_path = './results/intermediate/Places/' 
        for image_list_file in ['image_list_train.txt','image_list_test.txt'] :
            with open(label_path+image_list_file, 'r') as f:
                  image_list_all = f.readlines()
            CAM_map_all=[]    
            num_images = len(image_list_all)
            num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size_ori)))                
            for k in tqdm(range(0, num_batches)):
                    data_inp = image_list_all[k*batch_size_ori:min((k+1)*batch_size_ori,num_images)]
                    batch_size=len(data_inp)
                    input_img = torch.zeros(batch_size,3,224,224)
                    image_label=[]
                    for i,image_name_label in enumerate(data_inp):
                        image_name=image_name_label.split(' ')[0]
                        image_label.append(int(image_name_label.split(' ')[1][:-1]))
                        img = Image.open(image_path+image_name)       
                        img_RGB=img.convert('RGB')
                        input_img[i] = tf(img_RGB).unsqueeze(0)                        
                    conv_maps = conv_model(input_img.to(device))
                    conv_maps_np=conv_maps.cpu().detach().numpy()
                    if image_list_file=='image_list_train.txt':
                        for t in range(0,batch_size):
                            CAM_map= returnCAM(conv_maps_np[t], weight_softmax,image_label[t])
                            CAM_map_all.append(CAM_map)
                    if image_list_file=='image_list_test.txt':       
                        logits=linear_layer(nn.AvgPool2d(kernel_size=14)(conv_maps).view(-1, 512))
                        h_xs=F.softmax(logits,dim=1).cpu().detach()
                        for t in range(0,batch_size):
                            probs, idx = (h_xs[t]).sort(0, True)
                            CAM_map= returnCAM(conv_maps_np[t], weight_softmax,idx[0])
                            CAM_map_all.append(CAM_map)
            if image_list_file=='image_list_train.txt':
                np.save(result_path+'CAM_map_all_train.npy',CAM_map_all) 
            if image_list_file=='image_list_test.txt':       
                np.save(result_path+'CAM_map_all_test.npy',CAM_map_all)
