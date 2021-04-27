# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 18:24:43 2021

@author: SHIVAM SAURAV
"""
# -*- coding: utf-8 -*-


import numpy as np
import math
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy import stats
import gym
from gym import spaces
from stable_baselines.common.env_checker import check_env
from particle_filter import ParticleFilter
import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()


from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Unit import *
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Spice.Library import SpiceLibrary




import typing
import tqdm
from typing import Callable, List, Optional, Tuple, Union
from stable_baselines.common.vec_env import VecEnv

from matplotlib import colors


class custom_env(gym.Env) :

  
  def __init__(self,prior_mean,prior_var,threshold_var,observation_noise=1,slot_time=1,initial_condition=0,Voc=3.63,Rs=0.1,Rp=0.03,Cp=500,Rl=0):
    super(custom_env, self).__init__()
    self.Voc=Voc
    self.Rs=Rs
    self.Rp=Rp
    self.Cp=Cp
    #self.n=numb_of_exp
    self.Rl=Rl
    self.initial_condition=initial_condition
    #self.number_experiment=number_of_experiment
    self.Vc=initial_condition
    self.var_threshold=threshold_var
    self.ob=np.ones(30)
    self.ob.fill(0)
    self.slot_time=slot_time
    self.observation_noise=observation_noise#noise_var
    self.prior_mean=prior_mean
    self.prior_var=prior_var
    
    
    #self.observation=0#Vb


    # Define action and observation space
    # They must be gym.spaces objects
    
    #n_actions = 7
    #self.action_space = spaces.Discrete(n_actions)
    self.action_space = spaces.MultiDiscrete([7,5])  #current action ranges from 0 to 6 and frequency action ranges from 0 to 5
    #self.action_space =[Discrete(5), Discrete(3)]
    
    a=np.ones(30)
    a.fill(100)
    self.observation_space = spaces.Box(low=-a,high=a, shape=(30,))   #instead of space from -inf to +inf, we have taken it to be -100 to +100 on a (30,) vector
    #print(self.observation_space)                                    
    
  def reset_initial_condition(self): #reset_initial_conditions after every env.reset()
      self.initial_condition=0.0
    
      #for Particle Filter
  def pf(self,prior_var,number_particles,prior_Voc,prior_Rs,prior_Rp,prior_Cp,prior_Vc,observation_noise):
      self.test = ParticleFilter(prior_var=prior_var,number_particles=number_particles,prior_Voc=prior_Voc,prior_Rs=prior_Rs,prior_Rp=prior_Rp,prior_Cp=prior_Cp,prior_Vc=prior_Vc,observation_noise=observation_noise)
      
  
# =============================================================================
#   #KL(P||Q)
#   def kl_divergence(self,p, q):
#     epsilon=0.0000000001
#     p=np.where(p != 0, p , epsilon)
#     return np.sum(np.where(p != 0, p * np.log(p / q), 0)) #!np.all
# 
# #js divergence
#   def js_divergence(self,p, q):
#     m = (1./2.)*(p + q)
#     return (1./2.)*self.kl_divergence(p, m) + (1./2.)*self.kl_divergence(q, m)
# =============================================================================
        
  def reset(self):
      """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    
      self.reset_initial_condition()
      self.test.reset_prior(self.prior_var,prior_Voc=self.prior_mean[0],prior_Rs=self.prior_mean[1],prior_Rp=self.prior_mean[2],prior_Cp=self.prior_mean[3],prior_Vc=self.prior_mean[4])
      self.ob=np.ones(30)
      self.ob.fill(0) #on environment reset we start with Vb=0
      return np.array([self.ob]).astype(np.float32)
    

  def spice_circuit(self,amplitude=1,frequency=0,initial_condition=0):
    self.amplitude=amplitude
    self.frequency=frequency
    
    circuit = Circuit('Battery Equivalent model')

    circuit.V('OCV', 1, circuit.gnd, self.Voc@u_V) #name,positive node,negative node,value
    circuit.R('Rs', 1, 'cap_input_voltage' , self.Rs@u_Ω) #node 2='source_input_node' and node 3='source_output_node'
    circuit.R('Rp', 'cap_input_voltage' , 'cap_output_voltage', self.Rp@u_Ω)
    circuit.C('Cp', 'cap_input_voltage' , 'cap_output_voltage', self.Cp@u_F , ic=0@u_V) #ic=initial condition
    circuit.R('Rl', 'cap_output_voltage' , 'source_input_node', self.Rl@u_Ω)  #Rl=0
    #print(circuit)
    
    if self.frequency==0:
        circuit.I('current_source', 'source_input_node',circuit.gnd ,self.amplitude@u_A)
    else:
        circuit.SinusoidalCurrentSource('current_source', 'source_input_node',circuit.gnd ,0@u_A, offset=0@u_A,amplitude=self.amplitude@u_A , frequency=self.frequency@u_Hz)#freq=0.4


    #ac_line = circuit.AcLine('current_source', 'source_input_node','source_output_node' ,  frequenc=0.4u_Hz)
    
    #print(circuit)
    print(initial_condition)
    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    simulator.initial_condition(cap_input_voltage=initial_condition@u_V)
    analysis = simulator.transient(step_time=0.01@u_s, end_time=1@u_s) #@u_s is sec and @u_us is micro sec
    
    output_node=[]
    input_node=[]
    for node in (analysis['cap_output_voltage']): # 
       #print('Node {}: {} V'.format(str(node), float(node)))
        output_node.append(float(node))
   # print(output_node)
    
    for node in (analysis['cap_input_voltage']): # 
       #print('Node {}: {} V'.format(str(node), float(node)))
        input_node.append(float(node))
# =============================================================================
#     print("output_node - input_node : ",input_node[100]-output_node[100])
#     print("output_node  ",output_node[100])
#     print("input ",input_node[100])
#     print("output_node shape ",len(output_node))
# =============================================================================
    
# =============================================================================
#     figure1, ax = plt.subplots(figsize=(20, 10))
#     ax.set_title('Battery Equivalent Model')
#     ax.set_xlabel('Time [s]')
#     ax.set_ylabel('Voltage [V]')
#     ax.grid()
#     ax.plot(analysis.cap_input_voltage-analysis.cap_output_voltage)
#     #ax.plot(analysis.cap_input_voltage)
# =============================================================================
    Vb=output_node[100]
    Vb= np.random.normal(Vb, scale=math.sqrt(self.observation_noise))
    return(Vb,input_node[100]-output_node[100]) #at every 1 we are returning the output value 100*step_time=1sec,in array it will be 99th value100*step_time=0.1 sec,in array it will be 99th value



  #dk is the design variable in sOED literature which is action in RL terms and current for our experiment
  def step(self,action):
      
      
    #print(k," | mean : ",self.state[0]," | var : ",self.state[1])
    #print(action)
    if action[1]==0:
      self.freq=0.0004
    if action[1]==1:
      self.freq=0.004
    if action[1]==2:
      self.freq=0.04
    if action[1]==3:
      self.freq=0.4
    if action[1]==4:
      self.freq=4

    #self.freq=0
    self.dk=(action[0]+3)/10
    
    #print(self.orig_theta* self.dk) 
    
    #print("initial_condition: ",self.initial_condition)
    self.obs,cap_voltage=self.spice_circuit(self.dk,frequency=self.freq,initial_condition=self.initial_condition) #at every 0.1 sec
    self.initial_condition=cap_voltage
    '''
    Here,we take observation from our pyspice method(partially observable in terms of POMDP), we pass it to the Particle filter. We samples from the Particle filter.
    We require to create a mapping from these samples to a distribution which will be fed into the DQN module. However,we take a heuristic approach and get the mean vector & covariance matrix 
    from the samples. Create the 30 length vector containing first 25 elements as elements of covariance matrix and the other 5 elements as elements of mean vector.
    However,this module of Particle filter and creating 30 length vector should go in te Agent part with DQN module,but we try to incorporate this in the env itself in order to be consistent 
    with gym environment so that we can use different gym baselines.
    '''
    mean,cov,var=self.test.simulate(self.obs,amplitude=action[0],frequency=action[1],slot_time=1) 
    print("Mean Vector : ",mean)
    print("Variance Vector : ",var)
    #print(mean,cov)
    ####flattening row wise,replace 'C' with "F" to flatten columnwise
    self.ob=cov.flatten('C')
    self.ob=np.concatenate((self.ob,mean)) #feature vector containing 25 elements of covariance vector and 5 elements of mean vector
    
    
    
    done=np.all(np.less_equal(var,self.var_threshold))
    #print(done)
    #done=False
    
    
    #putting no of experiments as constraints instead of variance condition as done before 
    if done:
       # self.initial_condition=0.0
        reward=0
        # z = np.arange(-10, 10, 0.001)
        # posterior=norm.pdf(z, mean, math.sqrt(self.state[1]))   
        # reward=self.js_divergence(posterior,self.prior_dist)

    else:
        reward=-(0.7*1+0.3*self.dk**2) #70% weightage given to each iteration as cost and 30% to energy cost
      

    info={}
    #print(info)

    return np.array([self.ob]).astype(np.float32),reward,done,info

  def render(self):
    pass

  def close(self):
    pass



# =============================================================================
# 
# #with random agent
# 
# env = custom_env(number_of_experiment=5,initial_condition=0,Voc=3.63,Rs=0.1,Rp=0.03,Cp=500,Rl=0)
# 
# env.reset()
# #env.render()
# 
# # =============================================================================
# # print(env.observation_space)
# # print(env.action_space)
# # print(env.action_space.sample())
# # =============================================================================
# 
# action = 7
# #Random Agent
# n_steps = 1
# #n_steps = 2
# iterations=[0]
# for step in range(n_steps):
#   iter=0
#   #obs = env.reset()
#   env.reset()
#   done=False
#   # action, _ = model.predict(obs, deterministic=True)
#   # print("Step {}".format(step + 1))
#   # print("Action: ", action)
#   # obs, reward, done, info = env.step(action)
#   # print('obs=', obs, 'reward=', reward, 'done=', done)
#   # #env.render()
#   while not done:
#     iter=iter+1
#     print("Iteration no {}".format(iter))
#     #action, _ = model.predict(obs, deterministic=True)
#     print("Action: ", action)
#     obs, reward, done, info = env.step(action)
#     print('obs=', obs, 'reward=', reward, 'done=', done)
#     if done:
#       print("Goal reached!", "reward=", reward)
#       iterations.append(iter)
#       break
# print("Average iterations: ",np.mean(iterations))
# 
# =============================================================================
