# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 12:43:53 2021

@author: SHIVAM SAURAV

"""

from particle_filter import ParticleFilter
from pyspice_and_rL import custom_env
import numpy as np
from gym.spaces import Discrete, Tuple
from stable_baselines import DQN,ACKTR,A2C,PPO2


threshold_var=[0.1,0.1,0.1,0.1,0.1]
slot_time=1#in sec
observation_noise=1 #noise_var
prior_Voc=3.63
prior_Rs=0.02
prior_Rp=0.03
prior_Cp=500
prior_Vc=0
prior_mean=np.array([prior_Voc,prior_Rs,prior_Rp,prior_Cp,prior_Vc])
prior_var=np.array([1,1,1,1,1])
#for particle filter
number_particles=1000


env = custom_env(prior_mean,prior_var,threshold_var,observation_noise,slot_time,initial_condition=0,Voc=3.63,Rs=0.1,Rp=0.03,Cp=500,Rl=0)
env.pf(prior_var,number_particles,prior_Voc,prior_Rs,prior_Rp,prior_Cp,prior_Vc,observation_noise)
#model = A2C.load("A2C agent with pf")
model = A2C('MlpPolicy', env, verbose=1)
#env.reset()
#env.render()

# =============================================================================
# print(env.observation_space)
# print(env.action_space)
# print(env.action_space.sample())
# =============================================================================
#action =[0,0] #initial action
#Agent
n_steps = 2 #num_episodes
#var_threshold=[0.05,0.05,0.05,0.05,0.05] #Voc,rs,rp,cp
iterations=[]
for step in range(n_steps):
  iter=0
  obs = env.reset()
  
  done=False
  # action, _ = model.predict(obs, deterministic=False)
  print("Step {}".format(step + 1))
  # print("Action: ", action)
  # obs, reward, done, info = env.step(action.flatten())
  # print('obs=', obs, 'reward=', reward, 'done=', done)
  # # #env.render()
  while not done:
    iter=iter+1
    print("Iteration no {}".format(iter))
    action, _ = model.predict(obs, deterministic=False)
    obs, reward, done, info = env.step(action.flatten())
    #print('obs=', obs, 'reward=', reward, 'done=', done)
    
    # mean,cov,var=test.simulate(obs,amplitude=action[0],frequency=action[1],slot_time=1) ################### This we included in the environment code itself
    # print("Mean Vector : ",mean)
    # print("Variance Vector : ",var)
    # #print(mean,cov)
    # ####flattening row wise,replace 'C' with "F" to flatten columnwise
    # ob=cov.flatten('C')
    # ob=np.concatenate((ob,mean)) #feature vector containing 25 elements of covariance vector and 5 elements of mean vector
# =============================================================================
#     if (np.less_equal(var,var_threshold)):
#         done=True
# =============================================================================
    # done=np.all(np.less_equal(var,var_threshold))
    
    
    print("Action: ", action)
    #obs, reward, done, info = env.step(action)
    
    if done:
      print("Goal reached!", "reward=", reward)
      iterations.append(iter)
      break
print("Average iterations: ",np.mean(iterations))
