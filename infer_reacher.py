#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:04:33 2018

@author: shiyanpei
"""

from unityagents import UnityEnvironment
from agent_reacher import Agent
import torch
import numpy as np

env = UnityEnvironment(file_name='./Reacher20.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]



env_info = env.reset(train_mode=False)[brain_name] # reset the environment
state = env_info.vector_observations          # get the current state

num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
state_size = state.shape[1]

print('State size '+str(state_size))
print('Action size '+str(action_size))
print('Number of agents '+str(num_agents))

agent=Agent(state_size,action_size,0)
agent.actor_local.load_state_dict(torch.load('./ckpt/checkpoint1.pth'))

score = np.zeros(num_agents)                                       # initialize the score

while True:
    action = agent.act(state, add_noise=False)   # select an action     
    
    env_info = env.step(action)[brain_name]        # send the action to the environment                  
    next_state = env_info.vector_observations   # get the next state 
    reward = env_info.rewards                   # get the reward
    done = env_info.local_done                 # see if episode has finished
    
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if np.any(done):                                       # exit loop if episode finished
        break
    
print("Final Score: {}".format(np.mean(score)))

env.close()