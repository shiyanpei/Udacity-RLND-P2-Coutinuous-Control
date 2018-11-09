#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 15:03:10 2018

@author: shiyanpei
"""

from unityagents import UnityEnvironment
import numpy as np
from agent_single_arm import Agent
from collections import deque
import matplotlib.pyplot as plt
import torch

env = UnityEnvironment(file_name='./Reacher.app')
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

agent=Agent(33,4,1)
agent.actor_local.load_state_dict(torch.load('./ckpt/checkpoint1.pth'))

env_info = env.reset(train_mode=False)[brain_name]
state = env_info.vector_observations[0].reshape(1,-1)  
score=0

for j in range(2000):
    action = agent.act(state)
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0].reshape(1,-1)   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]             
        
    state = next_state
    if done:
        break 
    score += reward
print('total score is '+str(score))
env.close()