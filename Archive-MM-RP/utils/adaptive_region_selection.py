import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description = "Adaptive region selection")
parser.add_argument("-datasets", "--datasets", nargs='+',help="Specify the dataset", default=['Places','SUN397'])
parser.add_argument("-window_params", "--window_params",nargs='+',help="Specify the size and the stride for the sliding window to select discriminative patches", default=['3','1'])

args = parser.parse_args()

datasets=args.datasets
window_params=args.window_params


scales=['2','3']
win_size=window_params[0]
stride=window_params[1]


modes=['train','test']
win_size_half=int((win_size-1)/2)
for dataset in datasets:
    result_path='./results/intermediate/'+dataset+'/'    
    for mode in modes:
       CAM_map_all= np.load(result_path+'CAM_map_all_'+mode+'.npy')                     
       for scale in scales:        
            if scale=='2':
                scale_image_size=448
            if scale=='3': 
                scale_image_size=896
            local_maxima=[]                  
            for j in tqdm(range(len(CAM_map_all))):
                top_image=[]
                left_image=[]
                loc_m_image=[]
                CAM_map=CAM_map_all[j]
                ratio=scale_image_size/14                
                for a in range(1,13,stride):
                    for b in range(1,13,stride):
                        loc_m=CAM_map[a,b]              
                        if loc_m==np.max(CAM_map[a-win_size_half:a+win_size_half+1,b-win_size_half:b+win_size_half+1]):
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
                            top_image.append(top)
                            left_image.append(left)
                            loc_m_image.append(loc_m)
                local_maxima.append([top_image,left_image,loc_m_image])
            np.save(result_path+'local_max_'+scale+'_'+mode+'.npy',local_maxima)