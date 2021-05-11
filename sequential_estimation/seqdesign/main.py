# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 12:43:53 2021

@author: SHIVAM SAURAV

"""

#from particle_filter import ParticleFilter
#from pyspice_and_rL import custom_env
import numpy as np
from gym.spaces import Discrete, Tuple
from stable_baselines import DQN,ACKTR,A2C,PPO2
from tqdm import tqdm
from env import custom_env
from particlefilter import ParticleFilterRs
from particlefilter import ParticleFilterRp
from particlefilter import ParticleFilterVoc

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
threshold_var=[0.05,0.05,0.05,0.2,0.05] #Voc,rs,rp,tau
slot_time=1#in sec
observation_noise=0.0005 #noise_var
prior_Voc=3.63
prior_Rs=0.1
prior_Rp=0.03
prior_tau=15
prior_Vc=0

var_Voc=0.5
var_Rs=0.005
var_Rp=0.005
var_tau=0.5
var_Vc=0.005
# prior_mean=np.array([prior_Voc,prior_Rs,prior_Rp,prior_tau,prior_Vc])
# prior_var=np.array([0.5,0.005,0.005,0.5,0.005])
#for particle filter
number_particles=1000

#agent to compare with sequential estimation

first_current_profile=250
second_current_profile=1200 #end time of second profile
third_current_profile=1700 #end time of third profile
n_steps = first_current_profile+second_current_profile+third_current_profile
mean=[]

prior_mean=np.array([prior_Rs])
prior_var=np.array([var_Rs])

env = custom_env(prior_mean,prior_var,threshold_var,observation_noise,slot_time,initial_condition=0,Voc=0,Rs=0.1,Rp=0,Cp=0,Rl=0)
env.pfRs(prior_var,number_particles,prior_Voc,prior_Rs,prior_Rp,prior_tau,prior_Vc,observation_noise)

#test = ParticleFilterRs(prior_var=prior_var,number_particles=number_particles,prior_Voc=prior_Voc,prior_Rs=prior_Rs,prior_Rp=prior_Rp,prior_tau=prior_tau,prior_Vc=prior_Vc,observation_noise=observation_noise)
#test = ParticleFilter(number_timesteps=100,number_particles=100,prior_Voc=prior_Voc,prior_Rs=prior_Rs,prior_Rp=prior_Rp,prior_Cp=prior_Cp,prior_Vc=prior_Vc,observation_noise=observation_noise,prior_var=prior_var)

#env = custom_env(prior_mean,prior_var,threshold_var,observation_noise,slot_time,initial_condition=0,Voc=0,Rs=0.1,Rp=0,tau=1e-100,Rl=0)
#model = A2C.load("A2C agent with pf")
model = PPO2('MlpPolicy', env, verbose=1)
obs = env.reset()
for step in tqdm(range(first_current_profile)):
  obs, reward, done, info = env.step(0)
  mean.append(env.final_theta())

estimated_Rs=np.asarray(mean)[-1,0]

prior_mean=np.array([prior_Rp,prior_tau,prior_Vc])
prior_var=np.array([var_Rp,var_tau,var_Vc])

env = custom_env(prior_mean,prior_var,threshold_var,observation_noise,slot_time,initial_condition=0,Voc=0,Rs=0.1,Rp=0.03,Cp=500,Rl=0)
env.pfRp(prior_var,number_particles,prior_Voc,estimated_Rs,prior_Rp,prior_tau,prior_Vc,observation_noise)
#test = ParticleFilterRp(prior_var=prior_var,number_particles=number_particles,prior_Voc=prior_Voc,Rs=estimated_Rs,prior_Rp=prior_Rp,prior_tau=prior_tau,prior_Vc=prior_Vc,observation_noise=observation_noise)
#env = custom_env(prior_mean,prior_var,threshold_var,observation_noise,slot_time,initial_condition=0,Voc=0,Rs=0.1,Rp=0.03,tau=15,Rl=0)
model = PPO2('MlpPolicy', env, verbose=1)
obs = env.reset()
for step in tqdm(range(second_current_profile-first_current_profile)):
  obs, reward, done, info = env.step(1)
  mean.append(env.final_theta())

estimated_Rp=np.asarray(mean)[-1][0]
estimated_tau=np.asarray(mean)[-1][1]



prior_mean=np.array([prior_Voc,prior_Vc])
prior_var=np.array([var_Voc,var_Vc])

env = custom_env(prior_mean,prior_var,threshold_var,observation_noise,slot_time,initial_condition=0,Voc=3.63,Rs=0.1,Rp=0.03,Cp=500,Rl=0)
env.pfVoc(prior_var,number_particles,prior_Voc,estimated_Rs,estimated_Rp,estimated_tau,prior_Vc,observation_noise)
#test = ParticleFilterVoc(prior_var=prior_var,number_particles=number_particles,prior_Voc=prior_Voc,Rs=estimated_Rs,Rp=estimated_Rp,tau=estimated_tau,prior_Vc=prior_Vc,observation_noise=observation_noise)
#env = custom_env(prior_mean,prior_var,threshold_var,observation_noise,slot_time,initial_condition=0,Voc=3.63,Rs=0.1,Rp=0.03,tau=15,Rl=0)
model = PPO2('MlpPolicy', env, verbose=1)
obs = env.reset()
for step in tqdm(range(second_current_profile-first_current_profile)):
  obs, reward, done, info = env.step(2)
  mean.append(env.final_theta())

estimated_Voc=np.asarray(mean)[-1][0]
print("Mean Vector ",mean)

#print(np.asarray(mean)[:,0])
Rs_seq=np.asarray(mean)[0:first_current_profile]
Rp_seq = [item[0] for item in np.asarray(mean)[first_current_profile:second_current_profile]]
tau_seq=[item[1] for item in np.asarray(mean)[first_current_profile:second_current_profile]]
Voc_seq=[item[0] for item in np.asarray(mean)[second_current_profile:third_current_profile]]

#b=a.flatten()
print("Rs_seq ",Rs_seq)
print("Rp_seq ",Rp_seq)
print("tau_seq ",tau_seq)
print("Voc_seq ",Voc_seq)

# for step in tqdm(range(first_current_profile)):
#   obs, reward, done, info = env.step(0)
#   mean.append(env.final_theta())
  
  # done=False
  # if step<first_current_profile:
  #   obs, reward, done, info = env.step(0)
  #   mean.append(env.final_theta())
  # elif step<second_current_profile:
  #   obs, reward, done, info = env.step(1)
  #   mean.append(env.final_theta())
  # else:
  #   obs, reward, done, info = env.step(2)
  #   mean.append(env.final_theta())


  # action, _ = model.predict(obs, deterministic=False)
  #print("Step {}".format(step + 1))
  # print("Action: ", action)
  # obs, reward, done, info = env.step(action.flatten())
  # print('obs=', obs, 'reward=', reward, 'done=', done)
