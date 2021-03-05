#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
mu2=np.linspace(0.01,1, 20)
num_sample =3 
#pc 
mu2=mu2[0:num_sample]
num_time=3
time= np.linspace(0,num_time,num_time)


# In[30]:


params= np.zeros((len(time),2) )
params[:,0]= np.ones((len(time), 1)).reshape(-1,)
params[:,0]*=np.round(mu2[0],4)
params[:,1] = time


# In[33]:


for i in range(1, len(mu2)):
    tmp_ = np.zeros((len(time),2))
    tmp_[:,0]= np.ones((len(time), 1)).reshape(-1,)
    tmp_[:,0]*=np.round(mu2[i],4) # tmp_[:,0]= tmp_[:,0]*mu2[i]    
    tmp_[:,1] = time
    params = np.vstack((params,tmp_))
    
#print(params)
print(params.shape)


# In[49]:


import os
import numpy as np
snapshot = np.loadtxt('U1_1.csv',delimiter=",")


# In[61]:


#import os
#files=os.listdir(os.getcwd())
#print(files)
snapshot = np.loadtxt('U1_1.csv',delimiter=",")
print(snapshot.shape)

for i in range(1, num_sample):
    #testing
    # j=1
    # tmp_s= np.loadtxt("U1_{}.csv".format(j),delimiter=",")
    # real code
    tmp_s= np.loadtxt("U1_{}.csv".format(i+1),delimiter=",")
    snapshot= np.hstack((snapshot, tmp_s))
    print(snapshot.shape)
    print(tmp_s)
    del tmp_s


# In[48]:


print(snapshot)


# In[62]:


np.savetxt('SnapshotMatrix.csv',snapshot, delimiter=",")
np.savetxt('SnapshotMatrix.npy',snapshot, delimiter=",")
np.savetxt('params.csv',params, delimiter=",")
np.savetxt('params.npy',params, delimiter=",")


# In[ ]:




