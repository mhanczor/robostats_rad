#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:37:31 2017

@author: hades
"""

import gym, rb_gym
import matplotlib.pyplot as plt
plt.ion()

#env = gym.make('FrozenLake-v0')
#env = gym.make('LunarLander-v2')
env = gym.make('RadRoomSimple-v0')

env.reset()

for _ in range(100):
    obs, reward, done,_ = env.step(env.action_space.sample())
    env.render()
    plt.pause(0.01)
    if done:
        break

plt.pause(20)
