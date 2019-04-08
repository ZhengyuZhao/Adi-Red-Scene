import pandas as pd
import numpy as np
from tqdm import tqdm
label_path_Places='./datasets/labels/Places/'
label_path_SUN397='./datasets/labels/SUN397/'
# generate train-test split for Places365-Standard validation set 
pla_data = pd.read_csv(label_path_Places+'places365_val.txt', header=None, sep=' ')
pla_np = pla_data.values

count = 0
whole_train = []
whole_test = []
for i in range(365):
    temp = []
    for item in pla_np:
        if item[1] == i:
            temp.append(item)
        if len(temp) == 50:
            break
    whole_train.append(temp)

res_train = np.asarray(whole_train).reshape(-1, 2)
np.savetxt(label_path_Places+'image_list_train.txt', res_train,  fmt='%s')

res_test = np.asarray(pla_np)
to_remove = []

for item in tqdm(res_test):
    if item[0] not in res_train[:, 0]:
        to_remove.append(item)

len(to_remove)

res_final = np.asarray(to_remove)
res_final.shape
res_final = res_final[np.lexsort(res_final[:,].T)]

np.savetxt(label_path_Places+'image_list_test.txt', res_final,  fmt='%s')



#Find the overlapping classes between SUN397 and Places. Then following the statement in our
#paper, for the training images of SUN397 which have ground truth overlapped with Places,
#we used the ground truth for generating the Dis-Map.
with open(label_path_Places+'categories_places365.txt', 'r') as f:
          Places_class_labels = f.readlines()          
with open(label_path_SUN397+'ClassName.txt', 'r') as f:
          SUN_class_labels = f.readlines()
class_overlap=-np.ones(397,dtype=int)
for i, SUN_class_name in enumerate(SUN_class_labels):
    for j,Places_class_name in enumerate(Places_class_labels):
        if SUN_class_name[:-1] in Places_class_name.split(' ')[0]:
            class_overlap[i]=Places_class_name.split(' ')[1]
            break
class_overlap[143]=130 
np.save(label_path_SUN397+'Class_overlap.npy',class_overlap)


#Merge the 10 train-test splits into one list to avoid repeating
#operation on the same image in the following steps.
image_list_all=[]
splits=['01','02','03','04','05','06','07','08','09','10']
for split in splits:
  with open(label_path_SUN397+'Training_'+split+'.txt', 'r') as f1:
          image_list_all = image_list_all+f1.readlines()
  with open(label_path_SUN397+'Testing_'+split+'.txt', 'r') as f2:
          image_list_all = image_list_all+f2.readlines()

f1.close()
f2.close()  
image_list_all=list(dict.fromkeys(image_list_all))
with open(label_path_SUN397+'image_list_all.txt', 'w') as f3:
    for item in image_list_all:
        f3.write("%s" % item)
f3.close()


#save the index of the images of train/test splits in the whole list
modes=['train','test']    
for split in tqdm(splits):
    for mode in modes:
        if mode =='train':
             with open(label_path_SUN397+'Training_'+split+'.txt', 'r') as f:
                 image_list_split = f.readlines()
        if mode =='test':
             with open(label_path_SUN397+'Testing_'+split+'.txt', 'r') as f:
                 image_list_split = f.readlines()         
        split_index=[] 
        for j in range(len(image_list_split)):
            for i in range(len(image_list_all)) :                           
                 if image_list_all[i]==image_list_split[j]:
                     split_index.append(i)
                     break
        np.save(label_path_SUN397+'index_'+mode+'_'+split,split_index)

