#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:37:31 2017

@author: hades
"""

import gym, rb_gym

#env = gym.make('FrozenLake-v0')
#env = gym.make('LunarLander-v2')
env = gym.make('RadRoomSimple-v0')

env.reset()

for _ in range(10):
    obs, reward, done, prob = env.step(env.action_space.sample())
    env.render()