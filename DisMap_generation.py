"""
 * This script is written using python3.6 and modified based on the original code provided in
 * https://github.com/CSAILVision/places365/blob/master/run_placesCNN_unified.py
 
"""

import torch
from torch.autograd import Variable as V
from torchvision import transforms as trn
from torch.nn import functional as F
import numpy as np
import cv2
from PIL import Image
import os

def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data.cpu().numpy()))

def returnMap(feature_conv, weight_softmax, idx):
    #generate the Dis-Map with the size of 14X14
    nc, h, w = feature_conv.shape
    DisMap = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
    DisMap = DisMap.reshape(h, w)
    DisMap = DisMap - np.min (DisMap)
    DisMap = DisMap / np.max(DisMap)
    DisMap = np.uint8(255 * DisMap)
    return DisMap

def  returnTF ():
# load the image transformer
    tf = trn.Compose([
        trn.Resize((224,224)),
#        trn.Scale(256),
#        trn.CenterCrop(224),
        trn.ToTensor (),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf


def load_model():
    # this model has a last conv feature map as 14x14
    model_file = 'whole_wideresnet18_places365_python36.pth.tar'
   #download pre-trained Places-CNN model with wideresnet18 structure
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
    model = torch.load(model_file, map_location=lambda storage, loc: storage)
    model.eval()
    model._modules.get('layer4').register_forward_hook(hook_feature)
    return model


tf = returnTF() # image transformer

dir_in = '' #specify the local path of the dataset

with open('ClassName_SUN397.txt', 'r') as f:# load the label names of all the categories.
    #This text document can be downloaded from the folder named Sources
 class_labels = f.readlines()
model = load_model() 
nums=len(class_labels)
for i in range(0,nums) : 
       class_name=class_labels[i][:-1]
       tf = returnTF() # image transformer
       dir_out=dir_in+'/'+class_name
       class_in=dir_in+class_name 
       if not os.path.exists(dir_out):
         os.makedirs(dir_out)
       for image_name in os.listdir(class_in): 
          features_blobs = []
          params = list(model.parameters())
          img = Image.open(class_in+'/'+image_name) #input each image
          img=img.convert('RGB')
          input_img = V(tf(img).unsqueeze(0), volatile=True)
          logit = model.forward(input_img)
          h_x = F.softmax(logit).data.squeeze()
          probs, idx = h_x.sort(0, True)  
          weight_softmax = params[-2].data.numpy()
          weight_softmax[weight_softmax<0] = 0
          DisMap = returnMap(features_blobs[0], weight_softmax, i) 
          #specifically, we use the ground truth label i, as above line, to generate the Dis-Map for each training image,
          #while using predicted label, i.e., idx[0] for each test image, by using the following commented line.
          #DisMap = returnMap(features_blobs[0], weight_softmax, idx[0]) 
         
          
          cv2.imwrite(dir_out+'/'+image_name, DisMap[0]) #save the generated Dis-Maps into the folder dir_out
       print(i)
