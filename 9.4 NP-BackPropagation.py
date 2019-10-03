# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 15:21:27 2019

@author: Fernando
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

os.chdir(r"C:\Users\Fernando\Documents\STUFF\GALILEO\3-Machine Learning II")
tr = pd.read_csv('EntrenamientoPerceptron.csv')

def ReLu(z):
    a = np.maximum(z, 0)
    return a.reshape((-1))

def cost_der(outputs, y):
        return (y - outputs)

def ReLu_der(val):
    return np.where(val < 0, 0, 1)

def fwd(x, B, W, y):
#Forwards receives values x and returns a lists of the Activation and Z Values of each layer
    #z11 = np.matmul(W[0][0],np.transpose(x)) + B[0][0]
    #z12 = np.matmul(W[0][1],np.transpose(x)) + B[0][1]
    #Equivalent of prev 2 commented lines
    Z1 = np.matmul(np.transpose(W[0]),x)+np.transpose(B[0])  #Multiplies inputs x1 and x2 by the first layer
    A1 = ReLu(Z1) #ACtivates Z1 (Relu)
    
    Z2 = np.matmul(np.transpose(W[1]),A1)+np.transpose(B[1])  #
    A2 = ReLu(Z2)

    Z3 = np.matmul(W[2],A2)+np.transpose(B[2])  #
    A3 = ReLu(Z3)

    A = [A1, A2, A3] #All activations, layer by layer
    Z = [Z1, Z2, Z3]
    return A, Z #Returns list of Zs and Activations of each layer.

def bwd(A, Z, B, W ,lr = 0.05):
    E = [np.zeros(b.shape) for b in B] #Creates space to store errors
    E[-1] = cost_der(A[-1],y)*ReLu_der(Z[-1]) #Error en ultima capa
    
    grad_b = np.asarray([np.zeros(b.shape) for b in B])
    grad_w = np.asarray([np.zeros(w.shape) for w in W])
    
    grad_b[-1] = E[-1]
    grad_w[-1] = E[-1]*A[-2]
    
    E[-2] = np.multiply(np.matmul(W[-1].transpose(), E[-1]).reshape(-1), ReLu_der(Z[-2]))
    grad_b[-2] = E[-2]
    grad_w[-2] = E[-2]*A[-3]
    
    E[-3] = np.multiply(np.matmul(W[-2], E[-2].transpose()).reshape(-1), ReLu_der(Z[-3]))
    grad_b[-3] = E[-3]
    grad_w[-3] = E[-3]*ReLu(x)

    B[0] = B[0] - lr*grad_b[0].transpose()
    B[1] =B[1] - lr*grad_b[1].transpose()
    B[2] =B[2] - lr*grad_b[2].transpose()
    B
    W = W - lr*grad_w
    return B, W

def Init():
    #Hidden Layer 1: 2 Neurons
    nl1 = 2 
    b1 = np.random.normal(0,0.5,size=(nl1, 1)) #random initialization values for b1 (within 0 and 1)
    w1 = np.random.normal(0,0.5,size=(nl1 , 2)) #w1 has a row of weights for each neuron in layer1
    #w1 has 2 rows of 2 columns each. Each row belong to each neuron in l1 and 
    #each column belog to each input received.
    
    #Hidden Layer 2: 2 Neurons
    nl2 = 2
    b2 = np.random.normal(0,0.5,size=(nl2 , 1))
    w2 = np.random.normal(0,0.5,size=(nl2 , 2))
    
    #b2 = np.random.randn(nl2 , 1) #random initialization values for b2 (within 0 and 1)
    #w2 = np.random.randn(nl2 , 2)
    
    #Output Layer 3: 1 Neurons
    nl3 = 1
    b3 = np.random.normal(0,0.5,size=(nl3 , 1))
    w3 = np.random.normal(0,0.5,size=(nl3 , 2))
    
    #b3 = np.random.randn(nl3 , 1) #random initialization values for b2 (within 0 and 1)
    #w3 = np.random.randn(nl3 , 2)
    
    B = np.asarray([b1,b2,b3])
    W = np.asarray([w1,w2,w3])
    return B, W

#Input data in form of arrays (Layer = 0 consists of 2 input neurons)
x = np.asarray(tr.iloc[2, 0:2])
y = np.asarray(tr.iloc[2, 2])

B, W = Init()
#Forward propagating x1=12.104981 and x2=10.580729, which yields y = 1 (as result)
A, Z = fwd(x, B, W, y) #forwarding x which consists of an array with value x1 and x2 
B1, W1 = bwd(A, Z, B, W ,lr = 0.05) #Backpropagation #1

B, W = Init()
A1, Z1 = fwd(x, B1, W1, y) #2nd exp
B2, W2 = bwd(A1, Z1, B1, W1 ,lr = 0.05) #Backpropagation #2

B, W = Init()
A2, Z2 = fwd(x, B2, W2, y) #3rd exp
B3, W3 = bwd(A2, Z2, B2, W2 ,lr = 0.05) #Backpropagation #3

B, W = Init()
A3, Z3 = fwd(x, B3, W3, y) #4th exp
B4, W4 = bwd(A3, Z3, B3, W3 ,lr = 0.05) #Backpropagation #4

B, W = Init()
A4, Z4 = fwd(x, B4, W4, y) #5th exp
B5, W5 = bwd(A4, Z4, B4, W4 ,lr = 0.05) #Backpropagation #5

#Representacion intermedia de cada capa oculta.... todas se ven casi igual.
print(Z[1])
print(Z1[1])
print(Z2[1])
print(Z3[1])
print(Z4[1])

#Plotting each 2nd layer representation
fig = plt.figure()
ax1 = fig.add_subplot(111)
i = [1,2]
ax1.scatter(i, Z[1], s=10, c='b', marker="s", label='1st')
ax1.scatter(i, Z1[1], s=10, c='red', marker="s", label='2nd')
ax1.scatter(i, Z2[1], s=10, c='brown', marker="s", label='3rd')
ax1.scatter(i, Z3[1], s=10, c='green', marker="s", label='4th')
ax1.scatter(i, Z4[1], s=10, c='orange', marker="s", label='5th')
plt.ylim(-0.65, 0.70)
plt.xlim(-0.1, 2.15)
plt.legend(loc='upper left');
plt.show()

#CONCLUSION:
#Even when the weights and Biases are initialized randomly every time, by feeding the same inputs
#the neurons come up with the same output in the middle layer. This can be shown by printing the outputs as above
#or, by plotting each experiment representation as seen in the above graph.