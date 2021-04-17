# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 12:43:53 2021

@author: SHIVAM SAURAV

"""

from particle_filter import ParticleFilter
from pyspice_and_rL import custom_env
import numpy as np
from stable_baselines import DQN

# =============================================================================
# test = ParticleFilter(number_timesteps=100,number_particles=100)
# mean,cov=test.simulate(amplitude=1,frequency=1,slot_time=1)
# #print(mean,cov)
# ####flattening row wise,replace 'C' with "F" to flatten columnwise
# ob=cov.flatten('C')
# ob=np.concatenate((ob,mean))
# =============================================================================
#ob.append(posterior_mean)
#print(ob)



env = custom_env(number_of_experiment=5,initial_condition=0,Voc=3.63,Rs=0.1,Rp=0.03,Cp=500,Rl=0)
model = DQN('MlpPolicy', env, verbose=1)
env.reset()
#env.render()

# =============================================================================
# print(env.observation_space)
# print(env.action_space)
# print(env.action_space.sample())
# =============================================================================
test = ParticleFilter(number_timesteps=100,number_particles=100)
action = 0 #initial action
#Agent
n_steps = 2
#n_steps = 2
iterations=[0]
for step in range(n_steps):
  iter=0
  obs = env.reset()
  
  done=False
  #action, _ = model.predict(obs, deterministic=True)
  # print("Step {}".format(step + 1))
  # print("Action: ", action)
  # obs, reward, done, info = env.step(action)
  # print('obs=', obs, 'reward=', reward, 'done=', done)
  # #env.render()
  while not done:
    iter=iter+1
    print("Iteration no {}".format(iter))
    obs, reward, done, info = env.step(action)
    print('obs=', obs, 'reward=', reward, 'done=', done)
    
    mean,cov=test.simulate(obs,amplitude=1,frequency=1,slot_time=1)
    #print(mean,cov)
    ####flattening row wise,replace 'C' with "F" to flatten columnwise
    ob=cov.flatten('C')
    ob=np.concatenate((ob,mean)) #feature vector containing 25 elements of covariance vector and 5 elements of mean vector
    
    
    action, _ = model.predict(ob, deterministic=True)
    
    print("Action: ", action)
    #obs, reward, done, info = env.step(action)
    
    if done:
      print("Goal reached!", "reward=", reward)
      iterations.append(iter)
      break
print("Average iterations: ",np.mean(iterations))
