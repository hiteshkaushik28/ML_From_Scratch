#!/usr/bin/env python
# coding: utf-8

# ### <font color = "blue">Question 1) What are the number of parameters in 1st convolutional layers</font>

# #### Total Filters = 6
# #### Size of each Filter = 5 * 5 * 4 = 100
# #### Total Parameters = 6 * 100 = 600

# ### <font color = "blue">Question 2) What are the number of parameters in pooling operation?</font>

# #### Pooling operation doesn't involve any trainable parameters. It just outputs the maximum value of a region.
# #### Total Parameters = 0

# ### <font color = "blue">Question 3) Which of the following operations contain most number of parameters?</font>
# #### (a) conv 
# #### (b) pool
# #### (c) Fully connected layer (FC)
# #### (d) Activation Functions

# ### Parameters in each layer :
# #### Layer 1 : 600
# #### Layer 2 : 0
# #### Layer 3 : 2400 (16 filters each pf size 5 * 5 * 6)
# #### Layer 4 : 0
# #### Layer 5 : 48000 (120 filters each of size 5 * 5 * 6)
# #### Layer 6 : 10080 (84 filter of size 1 * 1 * 120)
# #### Layer 7 : 840 (Gaussian connection matrix of size 84 * 10)
# ### <font color = "blue">Answer : Maximum parameter estimation is required by Layer 5.</font>

# ### Question 4) <font color = "blue">Which operation consume most amount of memory?</font>
# #### (a) initial convolution layers
# #### (b) fully connected layers at the end
# 
# ### Answer : Fully connected layers at the end consumes most amount of memory as the number of trainable parameters added with memory required for output of those layers is maximum of all.

# In[ ]:




