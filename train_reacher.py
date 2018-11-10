#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 21:04:15 2018

@author: shiyanpei
"""

from unityagents import UnityEnvironment
import numpy as np
from agent_reacher import Agent
from collections import deque
import matplotlib.pyplot as plt
import torch
# select this option to load version 1 (with a single agent) of the environment
env = UnityEnvironment(file_name='./Reacher20.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]
print('State dimension is '+str(state_size))
print('Action size is '+str(action_size))
print('Number of agents is '+str(num_agents))


episodes = 500
agent=Agent(state_size,action_size,42)
all_scores = []
avg_scores_window = []
noise_damp = 0 
scores_window = deque(maxlen=100)
for e in range(1,episodes):
    #env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    
    
    scores=np.zeros(num_agents)
    while True:
        actions = agent.act(states, noise_damp)
        env_info = env.step(actions)[brain_name]          
        next_states = env_info.vector_observations
        rewards = env_info.rewards                     
        dones = env_info.local_done

        
        agent.step(states, actions, rewards, next_states, dones,num_updates=1)
        states = next_states
        
        
        scores+=rewards
        if np.any(dones):
            break
    avg_score = np.mean(scores)
    scores_window.append(avg_score)
    all_scores.append(avg_score)
    avg_scores_window.append(np.mean(scores_window))
    if e%10==0:
        print('Episode '+str(e)+' Average score: '+str(np.mean(scores_window)))
    if np.mean(scores_window)>=30:
        print('Environment solved with episode'+str(e)+' Score: '+str(np.mean(scores_window)))
        torch.save(agent.actor_local.state_dict(), './ckpt/checkpoint1.pth')
        break
plt.figure()
plt.xlabel('Episode')
plt.ylabel('Scores')
plt.plot(all_scores)
plt.savefig('./pictures/Reacher.png')
        
env.close()