# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import mixture
from mpl_toolkits.mplot3d import Axes3D

with open('dict_state.p', 'rb') as fp:
    data = pickle.load(fp)
    
print(len(data.keys()))

# Select key with the highest number of elements
max_temp = 0
for i in data.keys():
    max = len(data[i])
    if max > max_temp:
        lung = len(data[i])
        key_rec = i
        max_temp = max
print(lung)
print(key_rec)

# Select key with the highest Q-average
max_temp_Q = 0
for i in data.keys():
    max_Q = np.mean(data[key][:,[3]])
     if max_Q > max_temp_Q:
         key_rec = i
         max_temp_Q = max_Q
         lung = len(data[i])
print(key_rec)
print(max_temp_Q)
print(lung)

target=[]
for i in data.keys():
    target.append(list(data[i][:,[3]].reshape(len(data[i][:,[3]]))))

final_list = []
for item in target:
    for elem in item:
        final_list.append(elem)

plt.scatter(final_list, np.arange(len(final_list)))
plt.show()

  
# for this key (16, 17, 21, 0, -1, 0)   
key = (16, 17, 21, 0, -1, 0)
# actions
print(data[key][:,[0]].reshape(len(data[key][:,[0]])))
print(data[key])
# plot Q over phi
print(data[key][:,[2]].reshape(len(data[key][:,[2]])))
x1 = data[key][:,[2]].reshape(len(data[key][:,[2]]))
y = data[key][:,[3]].reshape(len(data[key][:,[3]]))
plt.scatter(x1, y)
print(y)

#plot Q over theta
x2 = data[key][:,[1]].reshape(len(data[key][:,[1]]))
y = data[key][:,[3]].reshape(len(data[key][:,[3]]))
plt.scatter(x1, x2, s=y)


fig = plt.figure()
ax = Axes3D(fig)

ax.scatter(x1, x2, y)
plt.show()



g = mixture.GMM(n_components=2)
g.fit(np.concatenate()) 

 