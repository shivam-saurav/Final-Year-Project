# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 22:01:18 2021

@author: SHIVAM SAURAV
"""

import numpy as np
import scipy.stats
from numpy.random import random

class ParticleFilter():
    def __init__(self,number_timesteps=100,number_particles=100,prior_Voc=3.63,prior_Rs=0.1,prior_Rp=0.03,prior_Cp=500,prior_Vc=0): #Voc=3.63,Rs=0.1,Rp=0.03,Cp=500,Rl=0
        self.number_timesteps=number_timesteps
        self.number_particles=number_particles

    # Parameters
        #N=number_timesteps
        # What are the parameters to be estimated? Voc, Rs, Rp, Cp, Vc0
        self.number_parameters = 5
        #self.Vb=observation
        self.prior_mean = np.array([prior_Voc,prior_Rs,prior_Rp,prior_Cp,prior_Vc])
        self.prior_covariance_matrix = np.eye(self.number_parameters)
        
        self.roundoff_epsilon = 1e-100
        
        self.voc_index = 0
        self.rs_index = 1
        self.rp_index = 2
        self.cp_index = 3
        self.vc_index = 4
    
        self.observation_noise_variance = 1
    
    # Update the voltage on the capacitor
    def update_vc(self,particles_value, amplitude, frequency,slot_time):
      tau=particles_value[self.rp_index]*particles_value[self.cp_index]#rp*cp
      const=amplitude*particles_value[self.rp_index]/(1+4*(np.pi**2)*(tau**2)*frequency**2)
      particles_value[self.vc_index]=const*(np.cos(2*np.pi*frequency*slot_time)+2*np.pi*frequency*np.sin(2*np.pi*frequency*slot_time)-np.exp(-slot_time/tau))+particles_value[self.vc_index]*np.exp(-slot_time/tau)
      return particles_value[self.vc_index]
    
    # Prediction step
    def predict_particles(self,particles_value, number_particles, amplitude, frequency,slot_time):
      for i in range(number_particles):
        self.particles_value[i, self.vc_index] = self.update_vc(particles_value[i], amplitude, frequency,slot_time)
    
    # Initialize particles
    def initialize_particles(self,number_particles, prior_mean, prior_covariance_matrix, number_parameters):
      self.particles_value = np.zeros((number_particles, number_parameters))
      self.particles_weight = np.zeros(number_particles)
    
      for i in range(number_particles):
        #particles_value[i, :] = np.matmul(np.random.randn(1, number_parameters), prior_covariance_matrix) + prior_mean
        self.particles_value[i, :] = np.random.multivariate_normal(prior_mean,prior_covariance_matrix)
        self.particles_weight[i] = scipy.stats.multivariate_normal(prior_mean, prior_covariance_matrix).pdf(self.particles_value[i, :])
    
      self.particles_weight = self.particles_weight / (np.sum(self.particles_weight) + self.roundoff_epsilon)
    
      return self.particles_value, self.particles_weight
    
    # Update step
    def update_particles(self,particles_value, particles_weight, current_observation_vb, amplitude, frequency, slot_time):
        self.particles_value=particles_value
        self.particles_weight=particles_weight
        for i in range(len(particles_weight)):
            
          voc = self.particles_value[i, self.voc_index]
          rs = self.particles_value[i, self.rs_index]
          vc = self.particles_value[i, self.vc_index]
          ib = amplitude * np.cos(2 * np.pi * frequency * slot_time)
          likelihood = scipy.stats.norm(voc - ib * rs - vc, self.observation_noise_variance).pdf(current_observation_vb)   
          self.particles_weight[i] = self.particles_weight[i] * likelihood
        
      
        self.particles_weight += self.roundoff_epsilon
        self.particles_weight /= np.sum(self.particles_weight)
        #print("########################",t,"####################")
        #print(particles_weight)
        
    
    def stratified_resample(self,weights):
        N = len(weights)
        # make N subdivisions, chose a random position within each one
        positions = (random(N) + range(N)) / N
    
        indexes = np.zeros(N, 'i')
        cumulative_sum = np.cumsum(weights)
        i, j = 0, 0
        while i < N:
            if positions[i] < cumulative_sum[j]:
                indexes[i] = j
                i += 1
            else:
                j += 1
        return indexes
    
    
    def resample_from_index(self,particles_value, particles_weight, indexes):
        self.particles_value=particles_value
        self.particles_weight=particles_weight
        self.particles_value[:] = self.particles_value[indexes]
        self.particles_weight.resize(len(self.particles_value))
        self.particles_weight.fill (1.0 / len(self.particles_weight))  ##should we give equal weights to resampled value??
        #return particles,weights
      
    def neff(self,weights):
        return 1. / np.sum(np.square(weights))
      
    
    def simulate(self,observation,amplitude=1,frequency=1,slot_time=1):
        # Simulation loop
# =============================================================================
#         amplitude = 1
#         frequency = 1
#         slot_time = 1
# =============================================================================
        
        self.particles_value, self.particles_weight = self.initialize_particles(self.number_particles, self.prior_mean, self.prior_covariance_matrix, self.number_parameters)
        
        
        for t in range(self.number_timesteps):
          #current_observation_vb = np.random.normal()
          self.predict_particles(self.particles_value, self.number_particles, amplitude, frequency,slot_time)
          self.update_particles(self.particles_value, self.particles_weight, observation, amplitude, frequency, slot_time) 
          #resample
          if self.neff(self.particles_weight) < len(self.particles_weight+1)/2: #N=length of Particles_weight
                    
              index=self.stratified_resample(self.particles_weight)
              #index=systematic_resample(particles_weight)
              self.resample_from_index(self.particles_value, self.particles_weight, index)
          posterior_mean,posterior_covariance,posterior_var=self.estimate(self.particles_value,self.particles_weight)
              
        return posterior_mean,posterior_covariance,posterior_var
      
        #print(self.particles_value)
        #print(self.particles_weight)
    
    def estimate(self,particles, weights):
      """returns mean and variance of the weighted particles"""
    
      particles_value=particles
      particles_weight=weights
      mean = np.average(particles_value, weights=particles_weight, axis=0)
      ''' For covariance : 
    x=[[-2.         -2.         -3.33333333]
     [ 1.          1.          1.66666667]]
    
    particles_weight = np.array([1,2])
    particle_weights=particle_weights.reshape(2,1)  : [[1]
                                                      [2]]
    
    
    particles_weight*X : element wise multiplication
    
    [[-2.         -2.         -3.33333333]
     [ 2.          2.          3.33333333]]
    
    '''
    
    #covariance formula : 1/sum(weights) * sum(X-mean)(X-mean)^T
      #print(particles_value.shape,mean.shape,particles_weight.shape)
      var  = np.average((particles_value - mean)**2, weights=particles_weight, axis=0)
      particles_weight=particles_weight.reshape(len(particles_weight),1)
      diff=particles_value-mean
      #print(diff)
      cov = 1./(particles_weight.sum())*np.dot((particles_weight*diff).T,diff)
      #print(sigma2)
      #var  = np.average((particles_value - mean)**2, weights=particles_weight, axis=0)
      return mean , cov ,var
    
    
    #print(posterior_mean)
    #print(posterior_covariance)
    
    
# =============================================================================
# test = ParticleFilter(number_timesteps=100,number_particles=100)
# mean,cov=test.simulate(amplitude=1,frequency=1,slot_time=1)
# print(mean,cov)
# =============================================================================
