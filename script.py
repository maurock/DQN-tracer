import pickle
import numpy as np

with open('Q_learning_table.p', 'rb') as fp:
    table = pickle.load(fp)
    
third_key = np.array([])
fourth_key = np.array([])
fifth_key = np.array([])
for key in table.keys():
    third_key = np.concatenate((third_key, np.array([key[3]])))
    fourth_key = np.concatenate((fourth_key, np.array([key[4]])))
    fifth_key = np.concatenate((fifth_key, np.array([key[5]])))

unique_third = np.unique(third_key)
unique_fourth= np.unique(fourth_key)
unique_fifth = np.unique(fifth_key)

for key in table.keys():
    print(table[key])
