#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipdb
from tabulate import tabulate
import time
import operator
import sys
from sklearn.metrics import accuracy_score
eps = np.finfo(float).eps
from itertools import combinations 


# In[10]:


def preprocess_data(path):
    data = pd.read_csv(path, delimiter = ';')
    return data


# In[44]:


# preprocess_data("/home/kaushik/Coursework/SEM2/SMAI/Assignments/Assignment_3/input/wine-quality/data.csv")


# In[11]:


def normalize(data):
    for i in data.columns[:-1]:
        data[i] = (data[i] - data[i].mean())/data[i].std()
    return data


# In[12]:


def split_data(data):
    train_data, val_data = np.split(data, [int(.8*len(data))])
    train_data = train_data.reset_index(drop = True)
    val_data = val_data.reset_index(drop = True)
    return train_data, val_data


# In[16]:


def input_output_split(data):
    input = data.iloc[:,0:11]
    input.insert(0,'ones',1)
    output = data.iloc[:,11]
    return input, output


# In[17]:


def sigmoid_func(input):
    result = 1 / (1 + np.exp(-input))
    return result


# In[22]:


def theta_gen(alpha, epochs, input, actual_output):
    theta = np.zeros(input.shape[1])
    for i in range(epochs):
        z = np.dot(input, theta)
        h = sigmoid_func(z)
        gradient = np.dot(input.T, (h - actual_output)) / actual_output.size
        theta -= alpha * gradient
    return theta


# In[37]:


def accuracy(predicted, test_data):
    print("Accuracy:", str(100 * np.mean(predicted == test_data)) + "%")


# In[38]:


def one_vs_all():
    alpha = 0.01
    epochs = 10000
    data = preprocess_data("/home/kaushik/Coursework/SEM2/SMAI/Assignments/Assignment_3/input/wine-quality/data.csv")
    data = normalize(data)
    train_data, val_data = split_data(data)
    train_in, train_out = input_output_split(train_data)
    val_in, val_out = input_output_split(val_data)
    num = 11
    
    classifiers = np.zeros(shape = (num, train_in.shape[1]))
    for c in range(0, num):
        label = (train_out == c).astype(int)
        classifiers[c, :] = theta_gen(alpha, epochs, train_in, label)
    
    temp = np.dot(val_in, classifiers.T)
    res = sigmoid_func(temp)
    prediction = res.argmax(axis = 1)
    return prediction, val_out


# In[39]:


predicted, test_data = one_vs_all()


# In[40]:


accuracy(predicted, test_data)


# In[93]:


# def one_vs_one():
#     alpha = 0.01
#     epochs = 10000
#     data = preprocess_data("/home/kaushik/Coursework/SEM2/SMAI/Assignments/Assignment_3/input/wine-quality/data.csv")
#     data = normalize(data)
#     train_data, val_data = split_data(data)
# #     train_in, train_out = input_output_split(train_data)
# #     val_in, val_out = input_output_split(val_data)
#     for i in range(11):
#         for j in range (i+1, 11):
#             x_train = train_data[]
            


# In[94]:


# one_vs_one()


# In[ ]:




