# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 10:07:02 2019
@author: Fernando
"""

import pandas as pd
import numpy as np
import os
import random

os.chdir(r"C:\Users\Fernando\Documents\STUFF\GALILEO\3-Machine Learning II")
tr = pd.read_csv('EntrenamientoPerceptron.csv')

P=tr['label'][tr['label']==1] #
N=tr['label'][tr['label']==0]
len(P)
len(N)
len(tr)

w = [ np.random.randn(),  np.random.randn()] #Random init of weights w
checker =0 #will count correct classifications as the while loop goes

print(f'w_init = {w}') #Prints startint point of Weights

while checker<100000:
    i = random.randrange(0, 199) #Picks a random set of x1 and x2 from df
    vals = np.asarray(tr.iloc[i,0:2])    #Converts to array to do matmul
    if tr['label'][i] == 1 and np.matmul(w,vals)<0: #If label=1 and w*vals is neg
        w += vals #Adds the x1,x2 to the weights (adjusts w)
    elif tr['label'][i] == 0 and np.matmul(w,vals)>=0:
        w -= vals #Substracts the x1,x2 from the weights (adjusts w)
    else:
        checker += 1 #Counts the correct validations
print(f'w_fin = {w}')

################ TESTING TRAINED WEIGHTS PRECISION ######################
def sigmoid(x):
  return 1/(1 + np.exp(-x))

w = np.transpose(w) #Transposing obtained w to do matmul
h = np.matmul(np.asarray(tr.iloc[:,0:2]), w) #Matrix multiplications to calculate all h values 
hs=sigmoid(h) #Activating h with Sigmoid function

accuracy = (200-sum(abs(tr['label'] - np.round(hs, 0))))/200
accuracy #Performed training with 100k correct clasifications.
