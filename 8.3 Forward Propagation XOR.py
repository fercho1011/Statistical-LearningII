# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 07:36:22 2019

@author: Fernando
"""

#Dado el diagrama de red neuronal de una capa oculta en el ejercicio de la diapositiva 
#"clase_MLP" implementar con numpy la etapa de forward propagation para los 4 posibles 
#puntos de la operacion Xor, usar matplotlib para gráficar el resultado de la capa 
#oculta y el Xor original(usando distinto color o marker para diferenciar puntos con y=1 
#de los puntos con y=0), comparar ambas gráficas y concluir/comentar resultados.
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1/(1 + np.exp(-x))

XORdom = np.asarray([[0, 0], [0,1], [1, 0], [1,1]]) #All possible inputs for XOR
w = np.asarray([[20, 20], [-20,-20]]) #weights
w = np.transpose(w) #Transposing to do matmul
b = np.asarray([-10,30]) #Biases
h = np.matmul(XORdom, w) + b #Matrix multiplications to calculate all h values 
hs=sigmoid(h) #Activating h with Sigmoid function

# Creating plot of intermediate h values (4 pairs)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(hs[:,0],hs[:,1], alpha=0.8, color=['salmon','brown','salmon', 'brown'], edgecolors='none', s=30)
plt.title('h values - FWD Prop XOR')
plt.legend(loc=2)
plt.show()

# Calculating output y 
w2 = np.asarray([20, 20]) #weights
w2 = np.transpose(w2) #Transposing to do matmul
b2 = np.asarray([-30]) #Biases

y = np.matmul(hs, w2) + b2
ys =sigmoid(y)
ys #XOR Values of each pair x1, x2 (XOR Domain)
np.round(ys, 0)
