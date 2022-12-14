import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

##accuracy funtion

def accuracy(dist):
    err=0
    right = []
    bad = 0
    ## for ANN 64 layers threshold = 0.37
    threshold = 0.37
    for i,z in enumerate(dist):
        if(z>threshold and i<anomaly):
            err=err+1
        if(z<threshold and i>anomaly):
            err=err+1
        if(z>threshold):
            right.append(0)
        else:
            right.append(1)
    print("Accuracy:",((len(dist)-err)/len(dist)))
    return right

X= np.load('FEATURE-TEST-COMP.npy')
anomaly=4922-3400

D_A = X[anomaly:,:]
len_X_test=D_A.shape
len_X_traning=X[:anomaly,:].shape
Lable_1=[1]
Lable_2=[0]
Label_1=[Lable_1 for i in range(len_X_traning[0])]
Label_2=[Lable_2 for i in range(len_X_test[0])]
Label_1=np.array(Label_1)
Label_2=np.array(Label_2)
Label = np.append(Label_1, Label_2, axis=0)
model = keras.models.load_model('ANN_64')
##P_train is the predicted value by trained model
P_train=model.predict(X)
dist=np.sqrt((P_train[:,0]-X[:,0])**2+(P_train[:,1]-X[:,1])**2 + (P_train[:,2]-X[:,2])**2)
print(dist.shape,P_train.shape)
##
dist=dist/np.max(dist)
plt.rcParams['figure.figsize']=20,10
plt.scatter(np.arange(2251),dist)
plt.scatter(anomaly,dist[anomaly],color='green')
print(np.mean(dist),np.var(dist))
print(np.mean(dist)+2*np.var(dist))

a=accuracy(dist)

b=[]
for i,j in enumerate(dist):
    if(i<anomaly):
        b.append(1)
    else:
        b.append(0)

print(len(a),len(b))

from sklearn.metrics import confusion_matrix
CM=confusion_matrix(b, a)
print(CM)

TN=CM[0,0]
TP=CM[1,1]
FP=CM[0,1]
FN=CM[1,0]
#calculate accuracy
FP,FN=FN,FP
accuracy=(TP+TN)/(TP+TN+FP+FN)
print(accuracy)
#calculate precision
precision=TP/(TP+FP)
print(precision)
#calculate recall
recall=TP/(TP+FN)
print(recall)
#calculate F1 score
F1=2* (recall*precision)/(recall+precision)
print(F1)