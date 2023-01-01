#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# In[76]:


import os 
path = os.listdir('C:/Users/Prajwal Kulkarni/MRI Scans/Training/')
classes = {'glioma_tumor':0, 'meningioma_tumor':1}


# In[77]:


pip install opencv-python


# In[78]:


import cv2
X = []
Y = []
for cls in classes:
    pth = 'C:/Users/Prajwal Kulkarni/MRI Scans/Training/'+cls
    for k in os.listdir(pth):
        img = cv2.imread(pth+'/'+k, 0)
        img = cv2.resize(img, (200,200))
        X.append(img)
        Y.append(classes[cls])


# In[79]:


X = np.array(X)
Y = np.array(Y)


# In[80]:


pd.Series(Y).value_counts()


# In[81]:


plt.imshow(X[0], cmap='gray');


# In[82]:


X.shape


# In[83]:


X_updated = X.reshape(len(X), -1)
X_updated.shape


# In[84]:


xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state = 10, test_size =.20)


# In[85]:


xtrain.shape, xtest.shape


# In[86]:


# FEATURE SCALING


# In[87]:


print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())


# In[88]:


# FEATURE SELECTION: PCA


# In[89]:


from sklearn.decomposition import PCA


# In[90]:


print(xtrain.shape, xtest.shape)
pca = PCA(.98)
pca_train = xtrain
pca_test = xtest


# In[91]:


# TO TRAIN MODEL


# In[92]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


# In[112]:


lg = LogisticRegression(C=0.1)
lg.fit(pca_train, ytrain)
import warnings
warnings.filterwarnings('ignore')


# In[113]:


sv = SVC()
sv.fit(pca_train, ytrain)


# In[114]:


#EVALUATION


# In[115]:


print("Training Score", lg.score(pca_train, ytrain))
print("Testing Score", lg.score(pca_test, ytest))


# In[116]:


print("Training Score", sv.score(pca_train, ytrain))
print("Testing Score", sv.score(pca_test, ytest))


# In[117]:


#PREDICTION


# In[118]:


prediction = sv.predict(pca_test)
np.where(ytest!=prediction)


# In[119]:


prediction[1]


# In[120]:


ytest[1]


# In[121]:


# TESTING MODEL


# In[122]:


dic = {0:'glioma_tumor', 1:'meningioma_tumor'}


# In[142]:


plt.figure(figsize=(12,8))
p = os.listdir('C:/Users/Prajwal Kulkarni/MRI Scans/Testing/')
c = 1
for m in os.listdir('C:/Users/Prajwal Kulkarni/MRI Scans/Testing/glioma_tumor/')[:12]:
    plt.subplot(3,4,c)
    img = cv2.imread('C:/Users/Prajwal Kulkarni/MRI Scans/Testing/glioma_tumor/'+m,0)
    img1 = cv2.resize(img, (200,200))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title("glioma_tumor")
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1


# In[ ]:




