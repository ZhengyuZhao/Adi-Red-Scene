import torch
from torchvision import transforms as trn
from torch.nn import functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import torchvision.models as models
from sklearn.preprocessing import Normalizer
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

arch='alexnet'
scales=['1','2','3']
#the threshold is set to 150 for both two local scales
T2=150
T3=150
pretrain_databases=['PL','PL','IN']
image_name='./examples/sun_ayzbohufbfsmzzye.jpg'

device = 'cpu'

def returnCAM(feature_conv, weight_softmax, class_idx):
#compute the Dis-Map with the size of 14x14  
    nc, h, w = feature_conv.shape
    cam = weight_softmax[class_idx].dot(feature_conv.reshape((nc, h*w)))
    cam = cam.reshape(h, w)    
    cam = cam - np.min (cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    return cam_img

def  returnTF (scale_image_size):
# load the image transformer
    tf = trn.Compose([
        trn.Resize((scale_image_size,scale_image_size)),
        trn.ToTensor (),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tf

def load_DisNet():
#Download the DisNet model, which is used for generating Dis-Map
    model_file = 'wideresnet18_places365.pth.tar'
    if not os.access(model_file, os.W_OK):
        os.system('wget http://places2.csail.mit.edu/models_places365/' + model_file)
        os.system('wget https://raw.githubusercontent.com/csailvision/places365/master/wideresnet.py')
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

def load_feature_extractor(scale,arch):
#load the pre-trained models for patch/image feature extraction,
#where for the global scale and local scale 1, the model that is
#pre-trained on Places2 database is used, while for the local scale 3, ImageNet was used.    
    if scale=='2' or scale=='1':
         # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
             weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
             os.system('wget ' + weight_url)        
        model = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict)
       
    if scale=='3':
        if arch=='resnet50':
          model=models.resnet50(pretrained=True)
        elif arch=='resnet18':
          model=models.resnet18(pretrained=True)
        elif arch=='alexnet':
          model=models.alexnet(pretrained=True)
    if arch=='resnet50' or arch=='resnet18':
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
    else:
        new_classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        model.classifier = new_classifier
        feature_extractor=model
    feature_extractor.eval()
    for param in feature_extractor.parameters():
            param.requires_grad = False
    return feature_extractor


tf=returnTF (224)
linear_layer,conv_model,weight_softmax = load_DisNet()
linear_layer=linear_layer.to(device)
conv_model=conv_model.to(device)

img = Image.open(image_name)  #load an image example     
input_img = tf(img).unsqueeze(0)
conv_maps = conv_model(input_img.to(device))
# Get 512 feature maps, each of which has a size of 14x14, in the last convolutional layer  
conv_maps_np=conv_maps.squeeze().cpu().clone().numpy()
logits=linear_layer(nn.AvgPool2d(kernel_size=14)(conv_maps).view(-1, 512))
h_x=F.softmax(logits,dim=1).cpu().clone().squeeze()
probs, idx = h_x.sort(0, True)

#Generate the Dis-Map for the example image, with top-1 prediction label
#. For training images,
#the ground truth label will be used here instead of idx[0].
CAM_map= returnCAM(conv_maps_np, weight_softmax,[203])#idx[0] is the top-1 prediction label output by the DisNet
#CAM_map= returnCAM(conv_maps_np, weight_softmax,[203])# 203 is the Ground truth label (kitchen) for the example image
#. For training images,the ground truth label will be used here instead of idx[0].

img_out = plt.imread(image_name)
height, width, _ = img_out.shape
heatmap = cv2.applyColorMap(cv2.resize(CAM_map,(224, 224)), cv2.COLORMAP_JET)
result = heatmap * 0.4 + cv2.resize(img_out,(224, 224)) * 0.5
#plt.imshow(np.uint8(result))
cv2.imwrite('./examples/image_heatmap.png',np.uint8(result))



#extract features from the whole image (global scale)
feature_extractor = load_feature_extractor('1',arch).to(device) 
feature_1 = feature_extractor(input_img.to(device)).cpu().detach().numpy() 
if arch == 'alexnet':
    fea_dim=4096
if arch == 'resnet18':
    fea_dim=512
if arch == 'resnet50':
    fea_dim=2048
for scale in ['2','3']:        
    if scale=='2':
        scale_image_size=448
        threshold=int(T2)
    if scale=='3': 
        scale_image_size=896
        threshold=int(T3) 
    ratio=scale_image_size/14 #Dis-Map is with a size of 14x14
    loc_image=[]
    for a in range(1,13): 
        for b in range(1,13):
            a_c=CAM_map[a,b]        
            #search for the discriminative regions that meet the requirements (as described in Section 3.2 of the paper).
            if a_c==np.max(CAM_map[a-1:a+2,b-1:b+2]) and a_c>threshold: 
                top=np.int32(a*ratio)-112
                left=np.int32(b*ratio)-112                   
                left=np.maximum(left,0)
                right=left+224       
                right=np.minimum(right,scale_image_size)
                left=right-224       
                top=np.maximum(top,0)
                down=top+224
                down=np.minimum(down,scale_image_size)
                top=down-224                 
                loc_image.append([top,left,a_c])
    feature_extractor = load_feature_extractor(scale,arch).to(device) 
    local_features = np.zeros(fea_dim)
    fig, ax = plt.subplots()
    img=img.resize((scale_image_size, scale_image_size))
    im = ax.imshow(img)
    ax.axis('off')
    for j in range(len(loc_image)):     
        ax.text(loc_image[j][1]+112,loc_image[j][0]+112, str(loc_image[j][2]),fontsize=20,color='g')
        p = patches.Rectangle((loc_image[j][1], loc_image[j][0]), 224, 224,linewidth=3,edgecolor='g',facecolor='none')
        ax.add_patch(p)
    plt.show()
    fig.savefig('./examples/image_patch_scale_'+scale+'.png', bbox_inches='tight')


    tf = returnTF(scale_image_size) # image transformer 

    for i in range(0,len(loc_image)):
        top=loc_image[i][0]
        left=loc_image[i][1]
        input_img=tf(img).unsqueeze(0)
        img_patch=input_img[:,:,top:top+224,left:left+224]
        patch_features = feature_extractor(img_patch.to(device)).cpu().detach().numpy()
        #max-pooling for intra-scale patch feature aggregation
        local_features=np.maximum(local_features,patch_features)            
    if scale=='2':           
        local_features_2=local_features
    if scale=='3':           
        local_features_3=local_features
        
feature_1=Normalizer().transform(feature_1)
local_features_2=Normalizer().transform(local_features_2)
local_features_3=Normalizer().transform(local_features_3)
#feature concatenation       
feature_concat=np.concatenate((feature_1,local_features_2,local_features_3),axis=1)

