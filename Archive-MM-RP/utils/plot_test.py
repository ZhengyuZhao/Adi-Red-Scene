#matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
final_path='./results/final/'
T2='150'
T3='100'




#Figure 3
result_path='./results/'+'intermediate/'+'Places/'
lm_2_train=np.load(result_path+'local_max_2_train.npy')
lm_2_test=np.load(result_path+'local_max_2_test.npy')
lm_3_train=np.load(result_path+'local_max_3_train.npy')
lm_3_test=np.load(result_path+'local_max_3_test.npy')
T = ['0', '50', '100', '150','200','225']
avg_patch=np.zeros(len(T),dtype=int)

for tt in range(len(lm_2_train)):
    print(tt)
    for num,t in enumerate(T):
        t=int(t)
        avg_patch[num] =avg_patch[num]+len(np.where(np.array(lm_2_train[tt][2]) > t)[0])+len(np.where(np.array(lm_2_test[tt][2]) > t)[0])+len(np.where(np.array(lm_3_train[tt][2]) > t)[0])+len(np.where(np.array(lm_3_test[tt][2]) > t)[0])
avg_dis=avg_patch/len(lm_2_train)/2/2            
y1=[]
x=[]
for i in T:
    y1.append(np.load(result_path+'acc_'+'1_2_3'+'_'+'alexnet'+'_'+'adi_red'+'_'+i+'_'+i+'_'+'PL'+'_'+'PL'+'_'+'IN'+'_'+'ori_res'+'.npy'))
    x.append(int(i))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y1,color='red', linewidth=3, marker='o')

ax.set_ylim(39, 43)#range of y-axis
ax.set_xlim(0, 250)#range of y-axis
for j in range(len(T)):     
    ax.text(x[j],y1[j]+0.3, str(avg_dis[j])[:4])
plt.xlabel('Threshold',fontsize=12,fontweight='semibold')
plt.ylabel('Classfication accuracy (%)',fontsize=12,fontweight='semibold')
ax.legend('2')
plt.show()
fig.savefig(final_path+'Figure3.png', bbox_inches='tight')
fig.clf()



#Figure 4

result_path='./results/'+'intermediate/'+'Places/'
lm_2_train=np.load(result_path+'local_max_2_train.npy')
lm_2_test=np.load(result_path+'local_max_2_test.npy')
lm_3_train=np.load(result_path+'local_max_3_train.npy')
lm_3_test=np.load(result_path+'local_max_3_test.npy')

avg_patch_img=np.zeros(len(lm_2_train),dtype=int)
for tt in range(len(lm_2_train)):
    print(tt)
    avg_patch_img[tt] =avg_patch_img[tt]+len(np.where(np.array(lm_2_train[tt][2]) > 150)[0])+len(np.where(np.array(lm_2_test[tt][2]) > 150)[0])+len(np.where(np.array(lm_3_train[tt][2]) > 100)[0])+len(np.where(np.array(lm_3_test[tt][2]) > 100)[0])
avg_dis=avg_patch/len(lm_2_train)/2/2            
avg_dis_class=np.zeros(365,dtype=float)
for i in range(0,len(lm_2_train),50):
    print(i)
    avg_dis_class[i//50]=np.mean(avg_dis[i:i+50])
    
label_path='./datasets/labels/'+'Places/'
class_sort_descend=np.argsort(avg_dis_class)[::-1]

with open(label_path+'categories_places365.txt', 'r') as f:
            class_names = f.readlines()
class_names_15_high=[class_names[i] for i in class_sort_descend[0:15]]
class_names_15_high=['/'.join(i.split(' ')[0].split('/')[2:]) for i in class_names_15_high]

class_names_15_low=[class_names[i] for i in class_sort_descend[350:365]]
class_names_15_low=['/'.join(i.split(' ')[0].split('/')[2:]) for i in class_names_15_low]

avg_dis_15_high=[avg_dis_class[i] for i in class_sort_descend[0:15]]

avg_dis_15_low=[avg_dis_class[i] for i in class_sort_descend[350:365]]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(class_names_15_high,avg_dis_15_high,color='green', linewidth=1,marker='.')
ax.plot(class_names_15_low,avg_dis_15_low,color='blue', linewidth=1,marker='.')
plt.xticks(rotation=270)
plt.xlabel('Scene categories',fontsize=8,fontweight='semibold')
plt.ylabel('Average number of discriminative regions per image',fontsize=8,fontweight='semibold')
plt.show()
fig.savefig(final_path+'Figure4.png', bbox_inches='tight')

fig.clf()




