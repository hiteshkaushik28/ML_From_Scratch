#!/usr/bin/env python
# coding: utf-8

# ## QUESTION 3 : Implement Linear Regression Model to Predict Chances of Admit

# ### <font color = "blue"> Import Required Modules

# In[50]:


import numpy as np
import sys
import math
from tabulate import tabulate
import pprint
import operator
import ipdb
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
eps = np.finfo(float).eps
from numpy import log2 as log


# ### <font color = "blue"> Utility to load and clean dataset

# In[51]:


def load_preprocess_data(path):
    dataset = pd.read_csv(path)
    dropIndex = [0]
    dataset.drop(dataset.columns[dropIndex],axis=1,inplace=True)
    return dataset


# ### <font color = "blue"> Utility to split data into training and validation using 80:20 ratio

# In[53]:


def split_data(dataset):
#     train_data = dataset.sample(frac=0.8)
#     val_data = dataset.drop(train_data.index)
    train_data, val_data = np.split(dataset, [int(.8*len(dataset))])
    return train_data, val_data


# ### <font color = "blue">Utility to normalise the dataset features

# In[54]:


def normalize(data):
    for i in data.columns[:-2]:
        data[i] = (data[i] - data[i].mean())/data[i].std()
    return data


# ### <font color = "blue">The Mean Square Error Loss Function

# In[59]:


def mean_square(X,y,theta):
#     tobesummed = np.power(((X @ theta.T)-y),2)
#     return np.sum(tobesummed)/(2 * len(X))
    return np.mean(((X @ theta.T)-y)**2)


# ### <font color = "blue"> Implementation of Gradient Descent - it is used to minimise the model loss.

# In[60]:


def gradientDescent(X,y,theta,iters,alpha, reg_param):
    for i in range(iters):
        gradient = np.sum(X * (X @ theta.T - y), axis=0) / len(X)
        theta[:, 0] -= (alpha * gradient[0])
        theta[:, 1 :] = (theta[:, 1:] * (1 - alpha * (reg_param / len(X)))) - alpha * gradient[1:]
    return theta


# ### <font color = "blue">Utility to set up required matrices for Linear Regression

# In[61]:


def setup_matrices(dataset):
    dataset = normalize(dataset)
    train_data, val_data = split_data(dataset)
    val_data.insert(0, 'Ones', 1)
    cols = train_data.shape[1]
    X = train_data.iloc[:, 0 : cols - 1]
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis=1)
    y = train_data.iloc[:, cols - 1 : cols].values
    theta = np.zeros([1, 8])
    return X, y, theta, val_data


# ### <font color = "blue">This function is used for plotting various graphs

# In[62]:


def plot_graph(iters, cost_matrix):
    fig, ax = plt.subplots()  
    plt.grid(True)
    ax.plot(np.arange(iters), cost_matrix, 'r')  
    ax.set_xlabel('Iterations')  
    ax.set_ylabel('Cost')  
    ax.set_title('Error vs. Training Epoch') 


# ### <font color = "blue"> Prediction function for validation or test dataset

# In[63]:


def predict(row, theta):
    length = theta.shape[1]
    value = 0
    for i in range(0, length):
        value += theta[0][i] * row[i]
    return value


# In[64]:


def data_validation(validate,theta):
    predicted = []
    actual = []
    for index, row in validate.iterrows():
        predicted.append(predict(row, theta))
        actual.append(row[-1])
    return predicted , actual


# In[65]:


def mean_squared_error(predicted , actual):
    return np.mean((np.array(actual) - np.array(predicted))**2)


# ### <font color = "blue"> Main Function 

# In[78]:


def main(reg_param):
    dataset = load_preprocess_data("../Input/AdmissionDataset/data.csv")
    alpha = 0.05
    iters = 250
    
    feature_matrix, actual_output, theta, val_data = setup_matrices(dataset)
    final_theta = gradientDescent(feature_matrix, actual_output, theta, iters, alpha, reg_param)
    train_error = mean_square(feature_matrix, actual_output, final_theta)
    predicted, actual = data_validation(val_data, final_theta)
    test_error = mean_squared_error(predicted, actual)
    return train_error, test_error


# In[79]:


def plot(xdata, y1data, y2data, title):
    fig, ax = plt.subplots(figsize=(8,6))  
    plt.title(title)
    ax.plot(xdata, y1data, color = "blue", label = "Train Error")
    ax.plot(xdata, y2data, color = "red", label = "Test Error") 
    plt.xlabel("Regularization Coefficient")
    plt.ylabel("Train /Test Error")
    plt.legend()
    plt.show()


# In[80]:


train_ridge_loss = []
test_ridge_loss = []
reg_list = []
for lam in range(0, 1000):
    train_error, test_error = main(lam)
    train_ridge_loss.append(train_error)
    test_ridge_loss.append(test_error)
    reg_list.append(lam)
plot(reg_list, train_ridge_loss, test_ridge_loss, "Ridge Regression")


# In[ ]:




