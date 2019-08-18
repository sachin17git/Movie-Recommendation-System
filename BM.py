#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 15:14:49 2019

@author: sachin
"""

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import pandas as pd

movies = pd.read_csv('ml-1m/movies.dat',
                     sep = '::',
                     header = None,
                     engine = "python",
                     encoding = "latin-1")

users = pd.read_csv('ml-1m/users.dat',
                     sep = '::',
                     header = None,
                     engine = "python",
                     encoding = "latin-1")

ratings = pd.read_csv('ml-1m/ratings.dat',
                     sep = '::',
                     header = None,
                     engine = "python",
                     encoding = "latin-1")

training_set = pd.read_csv('ml-100k/u1.base',
                           sep = '\t',
                           header = None)
training_set  = np.array(training_set,
                         dtype = "int")

test_set = pd.read_csv('ml-100k/u1.test',
                           sep = '\t',
                           header = None)
test_set  = np.array(test_set,
                     dtype = "int")

nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1]))) 


def convert(data):
    new_data = list()
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros((nb_movies))
        ratings[id_movies - 1] = id_ratings 
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)   
test_set = convert(test_set) 
        
# Creating Torch Tensors.
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1

test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
        
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        ph_given_v = torch.sigmoid(activation)
        return ph_given_v, torch.bernoulli(ph_given_v)
    
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        ph_given_h = torch.sigmoid(activation)
        return ph_given_h, torch.bernoulli(ph_given_h)
    
    def train(self, v0, vk, ph0, phk):
        self.W = self.W + (torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b = self.b + torch.sum((v0 - vk), 0)
        self.a = self.a + torch.sum((ph0 - phk), 0)
     
nv = len(training_set[0])    
nh = 100
batch_size = 100

rbm = RBM(nv, nh)

############### Training the Restricted Boltzmann Machine ##################

epochs = 10
for epoch in range(1, epochs + 1):
    train_loss = 0
    s = 0
    for id_user in range(0, nb_users - batch_size, batch_size):
        v0 = training_set[id_user: id_user + batch_size]
        vk = training_set[id_user: id_user + batch_size]
        ph0, _ = rbm.sample_h(v0)
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk, _ = rbm.sample_h(vk)     
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[vk >= 0]))
        s = s + 1
    print("Epoch: "+str(epoch)+" loss: "+str(train_loss/s))    
    
    
################# Testing the Restricted Boltzmann Machine ##################   
    
test_loss = 0
s = 0
for id_user in range(nb_users):
    v = training_set[id_user: id_user + 1]
    vt = test_set[id_user: id_user + 1]
    if(len(vt[vt >= 0])) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)  
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s = s + 1
print("Test loss: "+str(test_loss/s))      
                          
############ Manual predicting if user likes the movie or not #################

sample_loss = 0.
vv1 = training_set[2:3]
vtt = test_set[2:3]
_, hh = rbm.sample_h(vv1)
_, vv = rbm.sample_v(hh)
sample_loss += torch.mean(torch.abs(vtt[vtt >= 0] - vv[vtt >= 0]))
       
vv1 = vv1.numpy()         
vv = vv.numpy()        
vtt = vtt.numpy()
sample_loss = sample_loss.numpy()
        
        
        









