import sys

import matplotlib.pyplot as plt

import numpy as np
from six import StringIO, b
import math
from scipy.spatial import distance
from scipy.stats import norm as gaussian

#gym utils has seeding
from gym.utils import seeding
from gym import utils, spaces
import gym

#import Box2D
#from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, contactListener)
from particlefilter.ParticleFilter import ParticleFilter
import util as util


STEPSIZE = .5
PI = math.pi
MOVERR = STEPSIZE / 10 # Std Dev of the movements

class RadRoomSimple(gym.Env):

    def __init__(self, world_size=20, num_sources=3, strength=100, seed=None, vis=False, max_iters=300, map_sub=1):
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
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(world_size, world_size, 2))

        self.cov_thresh = 3.0 # point where we consider a fit guassian a prediction

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
#        print('resest')

        self.strengths += self.np_random.uniform(-10, 10, [self.num_sources,1])

        # Set the source locations [s,x,y]
        for i in range(self.num_sources):
            self.sources[i,0] = self.np_random.uniform(0, self.bounds[0])
            self.sources[i,1] = self.np_random.uniform(0, self.bounds[1])

        # RESET THE PARTICLE FILTER
        self.PF = ParticleFilter(min_particle_strength=0.5*np.min(self.strengths),
            max_particle_strength=1.5*np.max(self.strengths),
            num_sources=self.num_sources,
            world_size=self.bounds)

        heatmap = self.PF.get_heatmap(subsampling_factor=self.map_sub)
        obs = self.append_map(heatmap)

        obs = np.atleast_3d(obs)

        return obs

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
            self.heading += (45. + error * 5)
            self.loc += self.move(error)
        elif action == 2:
            # Rotate CW and drive forward
            self.heading -= (20. + error * 5)
            self.loc += self.move(error)
        elif action == 3:
            # Rotate CCW
            self.heading += (20. + error * 5)
        elif action == 4:
            # Rotate CW
            self.heading -= (20. + error * 5)

        # Confine the robot to the world bounds
        self.loc = np.clip(self.loc, 0, self.bounds)

        # Get a radiation reading from the sources
        self.get_reading()

        # Update the particle filter
        self.PF.step(self.reading, np.atleast_2d(self.loc))
        # Return the heatmap of particles
        heatmap = self.PF.get_heatmap(subsampling_factor=self.map_sub)
        obs = self.append_map(heatmap)

        # Get reward
        reward += -0.3 # Cost of living

        # If there is a high probability of the source being at a location
        # make a prediction, see how far it is from the nearest source ( there can only be one per source??) and then get a score based on that
        prediction_made = True
        means, covariances = self.PF.fit_gaussian(self.num_sources)
        if means is not None:
            for covar in covariances:
                v = np.linalg.eigvals(covar)
                v = 2 * np.sqrt(2*v)

                if np.max(v) > self.cov_thresh:
                    prediction_made = False
                    break

            # Make a prediction if all covariances are small, or we reached max_steps
            if prediction_made or self.steps > self.max_steps:
                # match each prediction with nearest source, without replacement
                pred_nn, source_nn = util.greedy_nearest_neighbor(means, self.sources)
                # compute distance
                dist = np.linalg.norm(pred_nn - source_nn, keepdims=True)
                # reward for each source decays with distance from source as a gaussian
                reward += 100 * gaussian(0,self.cov_thresh).pdf(dist) / self.num_sources
                # Stop running
                self.done = True

        obs = np.atleast_3d(obs)

        if type(reward).__module__ == np.__name__:
            reward = np.asscalar(reward)


        #obs = map? mean and xy location? (probably map since that will work best for A2C input)

        return obs, reward, self.done, {}


    def move(self, error):
        x = math.cos((PI/180.) * self.heading) * (STEPSIZE + error)
        y = math.sin((PI/180.) * self.heading) * (STEPSIZE + error)
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
        self.reading = np.round(np.sum(indv_reading)) # round to integer detection count

    def dist(self, p1, p2):
        return distance.cdist(p1, p2)

    def append_map(self, heatmap):
        # Create 'heatmap' of robot location
        loc_map = np.zeros_like(heatmap)
        int_location = np.floor(self.map_sub * self.loc).ravel().astype(np.int)
        if int_location[0] == self.bounds[0]: int_location -= 1
        if int_location[1] == self.bounds[1]: int_location -= 1
        loc_map[int_location[0], int_location[1]] = 1

        # Create Observation
        obs = np.stack((heatmap, loc_map), -1)
        return obs
