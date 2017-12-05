import sys

import matplotlib.pyplot as plt

import numpy as np
from six import StringIO, b
import math
from scipy.spatial import distance

#gym utils has seeding
from gym.utils import seeding
from gym import utils, spaces
import gym

#import Box2D
#from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
from particlefilter.ParticleFilter import ParticleFilter


STEPSIZE = 1
PI = 3.141592653589
MOVERR = 0.1 # Std Dev of the movements

class RadRoomSimple(gym.Env):

    def __init__(self, world_size=20, num_sources=1, strength=100, seed=None, vis=False, max_iters=100, map_sub=1):
        self._seed(seed)
        self.num_sources = num_sources
        self.vis = vis
        self.strengths = np.ones((num_sources, 1)) * strength
        self.bounds = np.array((world_size, world_size)) # min and max of the world
        self.max_steps = max_iters

        self.map_sub = map_sub # Subsampling factor for the heatmap

        self.sources = np.zeros((num_sources, 2)) # x-loc, y-loc
        self.rad_counts = 0.
        self.action_space = spaces.Discrete(4) # Forward, Diag-Left, Diag-Right, Rot-CCW, Rot-CW

    def _destroy(self):
        # This will reset the graphics or get rid of objects from the world
        raise NotImplementedError

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _reset(self):

        self.reading = 0. # radiation count reading
        self.steps = 0
        self.loc = np.zeros((1,2)) # X, Y location
        self.heading = 0. # Direction the bot is facing
        self.done = False

        self.strengths += self.np_random.uniform(-10, 10, self.num_sources)

        # Set the source locations [s,x,y]
        for i in range(self.num_sources):
            self.sources[i,0] = self.np_random.uniform(0, self.bounds[0])
            self.sources[i,1] = self.np_random.uniform(0, self.bounds[1])

        # RESET THE PARTICLE FILTER
        self.PF = ParticleFilter(min_particle_strength=np.min(self.strengths),
            max_particle_strength=np.max(self.strengths),
            num_sources=self.num_sources,
            world_size=self.bounds)

    def _step(self, action):

        reward = 0.
        self.steps += 1

        # Move the robot
        error = self.np_random.normal(0, MOVERR) # 0-mean error

        if action == 0:
            # Heading does not change
            self.loc += self.move(error)
        elif action == 1:
            # Rotate CCW and drive forward
            self.heading -= (45. + error * 5)
            self.loc += self.move(error)
        elif action == 2:
            # Rotate CW and drive forward
            self.heading += (20. + error * 5)
            self.loc += self.move(error)
        elif action == 3:
            # Rotate CCW
            self.heading -= (20. + error * 5)
        elif action == 4:
            # Rotate CW
            self.heading += (20. + error * 5)

        # Confine the robot to the world bounds
        np.clip(self.loc, 0, self.bounds)

        # Get a radiation reading from the sources
        self.get_reading()

        # Update the particle filter
        self.PF.step(self.reading, np.atleast_2d(self.loc))
        # Return the heatmap of particles
        heatmap = self.PF.get_heatmap(subsampling_factor=self.map_sub)

        # Get reward
        reward += -0.3 # Cost of living

        # If there is a high probability of the source being at a location
        # make a prediction, see how far it is from the nearest source ( there can only be one per source??) and then get a score based on that

        # if all the sources have been found, then done

        # if we've reached the iteration limit then done
        if self.steps > self.max_steps:
            #check our predictions and get the reward
            self.done = True

        obs = heatmap
        #obs = map? mean and xy location? (probably map since that will work best for A2C input)
        return obs, reward, self.done


    def move(self, error):
        x = math.cos((PI/180.) * self.heading) * STEPSIZE * error
        y = math.sin((PI/180.) * self.heading) * STEPSIZE * error
        return np.array((x, y))

    def generate_map(self):
        pass

    def render(self, mode='human', close=False):
        # Need to actually get the class instance to call render
        if close:
            plt.close()
        else:
            self.PF.render(sensor_location=self.loc, source_locations=self.sources)

    def get_reading(self):
        indv_reading = self.strengths / (1 + self.dist(self.sources, self.loc))
        self.reading = np.sum(indv_reading)

    def dist(self, p1, p2):
        return distance.cdist(p1, p2)
