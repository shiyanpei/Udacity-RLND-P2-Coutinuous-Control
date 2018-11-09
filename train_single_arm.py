#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:03:02 2018

@author: shiyanpei
"""

from unityagents import UnityEnvironment
import numpy as np
from agent_single_arm import Agent
from collections import deque
import matplotlib.pyplot as plt
import torch
# select this option to load version 1 (with a single agent) of the environment
env = UnityEnvironment(file_name='./Reacher.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
episodes = 1000
max_t=1000
agent=Agent(33,4,1)
scores=[]
scores_window = deque(maxlen=100)
for e in range(1,episodes):
    state = env_info.vector_observations[0].reshape(1,-1)
    
    
    score=0
    for t in range(max_t):
        action = agent.act(state, add_noise = False).reshape(1,-1)
        env_info = env.step(action)[brain_name]          
        next_state = env_info.vector_observations[0].reshape(1,-1)     
        reward = env_info.rewards[0]                      
        done = env_info.local_done[0] 
        
        
        agent.step(state, action, reward, next_state, done)
        state = next_state
        
        
        score+=reward
        if done:
            break
    scores.append(score)
    scores_window.append(score)
    if e%10==0:
        print('Episode '+str(e)+' Average score: '+str(np.mean(scores_window)))
    if np.mean(scores_window)>=30:
        print('Environment solved with episode'+str(e)+' Score: '+str(np.mean(scores_window)))
        torch.save(agent.actor_local.state_dict(), './ckpt/checkpoint1.pth')
        break
plt.figure()
plt.xlabel('Episode')
plt.ylabel('Scores')
plt.plot(scores)
plt.savefig('./pictures/single_arm.png')
        
env.close()