#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conservative Taylor Least Squares (TLS) computation on [-1,1]
@author: Elizaveta Wobbes
"""

import numpy as np  
from numpy.polynomial.legendre import leggauss
import matplotlib
import matplotlib. pyplot as plt

#create a class to store the input data: endpoints of the interval, locations of the particles (data points), and the function values at those locations  
class data_per_interval:
  def __init__(self, apoint, bpoint, particle_pos, particle_val):
    self.apoint = apoint
    self.bpoint = bpoint
    self.particle_pos = particle_pos
    self.particle_val = particle_val
  def get_apoint(self):
      return self.apoint
  def get_bpoint(self):
      return self.bpoint
  def get_particle_pos(self):
      return self.particle_pos
  def get_particle_val(self):
      return self.particle_val

#create the TLS class 
class TLS:
    def __init__(self, data_interval, evaluation_pos, conservation, conserve_value):
        self.data_interval  = data_interval
        self.evaluation_pos = evaluation_pos
        self.conservation   = conservation
        self.conserve_value = conserve_value   
    def get_data_interval(self):
        return self.data_interval
    def get_evaluation_pos(self):
        return self.evaluation_pos
    def get_conservation(self):
        return self.conservation
    def get_conserve_value(self):
        return self.conserve_value   
    def compute_Taylor_basis(self,a, b, x, cons):
        #we use only 3 Taylor basis functions in this case
        x_c   = (b+a)/2
        dx    = (b-a)/2
        psi_2 = (x-x_c)/dx
        psi_3 = (x-x_c)**2/(2*dx**2) - (1/3*b**3 - x_c*b**2 + x_c**2*b  - 1/3*a**3 + x_c*a**2 - x_c**2*a)/(b-a)/(2*dx**2)
        if cons == False:
            psi_1 = np.ones(len(x))
            psi   = (psi_1[np.newaxis],psi_2[np.newaxis],psi_3[np.newaxis])
        else:
            #psi_1 is redundant, if a_1 is provided
            psi = (psi_2[np.newaxis],psi_3[np.newaxis])
        return psi
    def compute(self):
        #we use the same notation as in the paper
        a    = self.get_data_interval().get_apoint()
        b    = self.get_data_interval().get_bpoint()
        x1   = self.get_data_interval().get_particle_pos()
        cons = self.get_conservation()
        #compute Taylor basis functions at particle positions
        B    = self.compute_Taylor_basis(a,b,x1,cons)
        B    = np.concatenate(B,axis=0)
        sB   = np.shape(B)
        D    = np.zeros([sB[0],sB[0]])
        for i in range(0,len(x1)):
            D = D + B[:,i][np.newaxis]*B[:,i][np.newaxis].T
        U   = self.get_data_interval().get_particle_val()
        U   = U[:,np.newaxis]
        x2  = self.get_evaluation_pos()
        #compute Taylor basis functions at Gauss points' positions
        Psi = self.compute_Taylor_basis(a,b,x2,cons)
        Psi = np.concatenate(Psi, axis=0).T
        if self.get_conservation() == True:
            U_modified = U-self.get_conserve_value()*np.ones([len(x1),1])
            BU = np.dot(B,U_modified)
        else:
            BU = np.dot(B,U)
        alpha = np.linalg.solve(D,BU)
        if self.get_conservation() == True:
            reconstruction = np.dot(Psi,alpha) + self.get_conserve_value()*np.ones([len(x2),1])
        else: 
            reconstruction = np.dot(Psi,alpha)
        return reconstruction
    
#example: use 3 particles within [-1,1] and reconstruct sin(x)+2
ploc1  = np.array((-0.9,0,0.5))
input1 = data_per_interval(-1,1,ploc1,np.sin(ploc1)+2)

#define evaluation points 
x_gauss, w_gauss = leggauss(2)

#the exact value of the integral, in this case, is equal to 4 -> a_1 = 2
#setting conservation to False gives the computation without a_1 = 2
input2 = TLS(input1, x_gauss, True, 2)
TLS_reconstruction = input2.compute()    
TLS_integral =  sum(w_gauss[:,np.newaxis]*TLS_reconstruction)   
print(TLS_integral)

#visualisation
x_plot = np.linspace(input1.apoint, input1.bpoint, 201)
plt.plot(input1.particle_pos, input1.particle_val, 'o', markerfacecolor = 'white', markeredgecolor = 'blue', label='known values')
plt.plot(input1.particle_pos, 0*input1.particle_val, 'o', markerfacecolor = 'blue', markeredgecolor = 'blue', label='data points')
plt.plot(x_gauss, TLS_reconstruction, 'o', markerfacecolor = 'white', markeredgecolor = 'red', label='TLS')
plt.plot(x_gauss, 0*TLS_reconstruction, 'o', markerfacecolor = 'red', markeredgecolor = 'red', label='Gauss points')
plt.plot(x_plot, np.sin(x_plot)+2, 'b-', label = 'sin(x)+2')
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title('TLS function reconstruction')
plt.legend()
plt.show()
