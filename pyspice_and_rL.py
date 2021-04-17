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

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()


from PySpice.Probe.Plot import plot
from PySpice.Spice.Netlist import Circuit
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from PySpice.Unit import *
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Spice.Library import SpiceLibrary


#taken from https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/common/evaluation.html#evaluate_policy and made few changes.

import typing
import tqdm
from typing import Callable, List, Optional, Tuple, Union
from stable_baselines.common.vec_env import VecEnv

from matplotlib import colors

class custom_env(gym.Env) :

  
  def __init__(self,number_of_experiment=10,initial_condition=0,Voc=3.63,Rs=0.1,Rp=0.03,Cp=500,Rl=0):
    super(custom_env, self).__init__()
    
    self.Voc=Voc
    self.Rs=Rs
    self.Rp=Rp
    self.Cp=Cp
    #self.n=numb_of_exp
    self.Rl=Rl
    self.initial_condition=initial_condition
    self.number_experiment=number_of_experiment
    #self.observation=0#Vb
# =============================================================================
#     z = np.arange(-10, 10, 0.001)
#     self.prior_dist=norm.pdf(z, self.state[0], math.sqrt(self.state[1]))
#     self.initial_condition=0.0
#     self.threshold_variance=threshold_var
#     self.cost=0.0
#     self.total_cost=[]
# =============================================================================

    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions, we have two: left and right
    n_actions = 7
    self.action_space = spaces.Discrete(n_actions)
    #self.action_space = spaces.Box(low=1,high=n_actions,shape=(1,),dtype=np.float32)
    #high=np.array([100,100])
    #hig=100
    a=np.ones(30)
    a.fill(100)
    self.observation_space = spaces.Box(low=-a,high=a, shape=(30,))   
    #print(self.observation_space)                                    
    
  def reset_initial_condition(self):
      #self.initial_condition=0
      self.number_experiment=self.number_experiment
        
  def reset(self):
      """
    Important: the observation must be a numpy array
    :return: (np.array) 
    """
    
    # Initialize the agent at the right of the grid
    #self.state = self.observation
    
    
    # here we convert to float32 to make it more general (in case we want to use continuous actions)
    
    #return self.initial_condition
      self.reset_initial_condition()
      self.observation=0
      return np.array([self.observation]).astype(np.float32)
    

  def spice_circuit(self,amplitude=1,frequency=0,initial_condition=0):
    self.amplitude=amplitude
    self.frequency=frequency
    
    circuit = Circuit('Battery Equivalent model')

    circuit.V('OCV', 1, circuit.gnd, self.Voc@u_V) #name,positive node,negative node,value
    circuit.R('Rs', 1, 'cap_input_voltage' , self.Rs@u_Ω) #node 2='source_input_node' and node 3='source_output_node'
    circuit.R('Rp', 'cap_input_voltage' , 'cap_output_voltage', self.Rp@u_Ω)
    circuit.C('Cp', 'cap_input_voltage' , 'cap_output_voltage', self.Cp@u_F , ic=0@u_V) #ic=initial condition
    circuit.R('Rl', 'cap_output_voltage' , 'source_input_node', self.Rl@u_Ω)
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
    analysis = simulator.transient(step_time=0.01@u_s, end_time=100@u_s)
    
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
    
    figure1, ax = plt.subplots(figsize=(20, 10))
    ax.set_title('Battery Equivalent Model')
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [V]')
    ax.grid()
    ax.plot(analysis.cap_input_voltage-analysis.cap_output_voltage)
    #ax.plot(analysis.cap_input_voltage)
    
    return(output_node[10000],input_node[10000]-output_node[10000]) #at every 0.1 millsec we are returning the output value 100*step_time=0.1 millisec,in array it will be 99th value100*step_time=0.1 sec,in array it will be 99th value

  
  

#KL(P||Q)
  def kl_divergence(self,p, q):
    epsilon=0.0000000001
    p=np.where(p != 0, p , epsilon)
    return np.sum(np.where(p != 0, p * np.log(p / q), 0)) #!np.all

#js divergence
  def js_divergence(self,p, q):
    m = (1./2.)*(p + q)
    return (1./2.)*self.kl_divergence(p, m) + (1./2.)*self.kl_divergence(q, m)

  #dk is the design variable in sOED literature which is action in RL terms and current for our experiment
  def step(self,action):

    #print(k," | mean : ",self.state[0]," | var : ",self.state[1])
    self.dk=(action+3)/10
    #print(self.orig_theta* self.dk) 
    
    #print("initial_condition: ",self.initial_condition)
    self.obs,cap_voltage=self.spice_circuit(self.dk,frequency=4,initial_condition=self.initial_condition) #at every 0.1 sec
    self.initial_condition=cap_voltage
    

    self.number_experiment=self.number_experiment-1
    #print(self.number_experiment)
    done=bool(self.number_experiment<2)
    #print(done)
    #done=False

    if done:
        self.initial_condition=0.0
        reward=0

    else:
        reward=-(0.7*1+0.3*self.dk**2) #70% weightage given to each iteration as cost and 30% to energy cost
      #self.cost=self.cost+self.dk**2 #energy

    info={}
    #print(info)

    return np.array([self.obs]).astype(np.float32),reward,done,info

  def render(self):
    pass

  def close(self):
    pass

  def total_costs(self):
    return(self.total_cost)

  def final_theta(self):
    return(self.state[0])

#env = custom_env()
# If the environment don't follow the interface, an error will be thrown
#check_env(env, warn=True)

# =============================================================================
# from stable_baselines import DQN, PPO2, A2C, ACKTR
# from stable_baselines.common.cmd_util import make_vec_env
# 
# # Instantiate the env
# env = custom_env(number_of_experiment=10,initial_condition=0,Voc=3.63,Rs=0.1,Rp=0.03,Cp=500,Rl=0)
# # wrap it
# env = make_vec_env(lambda: env, n_envs=1)
# =============================================================================

# agent   #verbose=1 means info about training
#model = DQN('MlpPolicy', env, verbose=1)#MlpPolicy,CnnPolicy,MlpLnLstmPolicy,MlpLstmPolicy,CnnLstmPolicy,CnnPolicy,CnnLnLstmPolicy
# obs=env.reset()
# action, _ = model.predict(obs, deterministic=False)
# print(action)

# =============================================================================
# mean_reward, std_reward,episode_rewards,total_iteration_in_each_episode = evaluate_policy(model, env, n_eval_episodes=10,deterministic=False)
# #before learning
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
# print("iteration per episodes",np.mean(total_iteration_in_each_episode))
# #plotcurve(episode_rewards)
# =============================================================================


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
