#!/usr/bin/env python
# coding: utf-8

# ##  One vs One approach

# In[150]:


import pandas as pd
import numpy as np
import math
import operator
from numpy import log2 as log
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib import pyplot as plt
eps = np.finfo(float).eps

df = pd.read_csv("../Input Data/wine-quality/data.csv")
df = df.sample(frac=1).reset_index(drop=True)
train_data, val_data = np.split(df,[int(0.8*len(df))])
threshold = 0.5 # consider this as standard value of threshold


# In[188]:


def initialize_train(data, label1, label2):
    theta = np.zeros((data.shape[1], 1))
    data = data.loc[data['quality'].isin([label1, label2])]
    X = (data.iloc[:,:-1] - data.iloc[:,:-1].mean()) / data.iloc[:,:-1].std()
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = data.iloc[:,-1]
    y = np.where(y == label1, 1, 0)
    y = y[:, np.newaxis]
    return X, y, theta


# In[152]:


def initialize_test(data):
    X = (data.iloc[:,:-1] - data.iloc[:,:-1].mean()) / data.iloc[:,:-1].std()
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = data.iloc[:,-1]
    y = y[:, np.newaxis]
    return X, y


# In[143]:


def sigmoid(theta, X):
    # Activation function used to map any real value between 0 and 1
    val = 1 / (1 + np.exp(-np.dot(X, theta)))
    return val


# In[144]:


def cost_function(theta, X, y):
    # Computes the cost function for all the training samples
    m = X.shape[0]
    sig_val = sigmoid(theta, X)
    total_cost = -(1 / m) * np.sum(y * np.log(sig_val) + (1 - y) * np.log(1 - sig_val))
    return total_cost


# In[145]:


def gradient(theta, X, y, change = 0.001, alpha = 0.001):
    # Computes the gradient of the cost function at the point theta
    m = X.shape[0]
    cost_change = 1
    iterations = 1
    cost = cost_function(theta, X, y)
    
    while cost_change > change:
        old_cost = cost
        temp = (alpha) * np.dot((sigmoid(theta,X) - y).T, X)
        theta = theta - temp.T
        cost = cost_function(theta, X, y)
        cost_change = old_cost - cost
        iterations += 1
        
    return theta, iterations


# In[199]:


def predict(theta, X, label1, label2):
    predicted_probab = sigmoid(theta, X)
    predicted_val = np.where(predicted_probab >= 0.5, 1, 0)
    return predicted_val[0]


# In[209]:


def all_thetas():
    all_set = {}
    for i in range(3,9): #consider i as positive if probab >= 0.5
        for j in range(i+1, 10):
            X, y, theta = initialize_train(train_data, i, j)
            theta, iterations = gradient(theta, X, y)
            all_set[str(i)+'-'+str(j)] = theta
    return all_set


# In[216]:


def logistic():
    val_x, val_y = initialize_test(val_data)
    Thetas = all_thetas()
    my_pred = []
    for row in val_x:
        max_class = {}    
        for theta in Thetas:
            k = theta.split('-')
            i = int(k[0])
            j = int(k[1])
            predicted = predict(Thetas[theta], row, i, j)
            if predicted == 1:
                predicted = i
            else:
                predicted = j
            if predicted  not in max_class:
                max_class[predicted] = 1
            else:
                max_class[predicted] += 1
        sorted_votes = sorted(max_class.items(), key=operator.itemgetter(1), reverse = True)
        my_pred.append(sorted_votes[0][0])
    
    actual = np.squeeze(val_y)
    confusion_mat = confusion_matrix(actual,my_pred)
    print(confusion_mat)
    print(accuracy_score(actual, my_pred)*100)


# In[217]:


logistic()


# In[218]:


df['quality'].unique()


# In[ ]:




