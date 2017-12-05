import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from scipy.stats import poisson

import ParticleFilter

# import ipdb as pdb
import pdb

plt.ion()

# World config
np.random.seed(42)
h = 20
w = 20
world_size = np.array([w,h]) # Note we consistently use w,h and x,y

num_particles = 5000
num_sources = 3

fusion_range = 5

# source_strength = 2.2e6 * 100 # 100 u Curies in CPM
source_strength = 100
min_particle_strength = 0.5*source_strength
max_particle_strength = 1.5*source_strength

# Initialize world
source_locations = world_size * np.random.rand(num_sources, 2)
source_strengths = source_strength * np.ones(num_sources)

sensor_location = np.array([[1.,2.]])

particle_filter = ParticleFilter.ParticleFilter(
    min_particle_strength,
    max_particle_strength,
    num_sources,
    world_size=world_size,
    num_particles=num_particles,
    fusion_range=fusion_range)

# particle_filter.render(sensor_location, source_locations)
# plt.pause(1)

# Take a reading
for t in range(300):
    source_dist_sq = distance.cdist(sensor_location, source_locations, 'sqeuclidean')
    source_expected_intensity = source_strengths / (1 + source_dist_sq)
    # reading = np
    reading = np.round(np.sum(source_expected_intensity))

    particle_filter.step(reading, sensor_location)
    # particle_filter.render(sensor_location, source_locations)
    # plt.pause(.01)

    # sensor_location += [0.06, 0.05+np.cos(t)/2]
    # sensor_location = np.random.randint(low=0, high=w, size=[1,2])

    # Choose next location weighted by particle density
    subsampling_factor = 2
    heatmap = particle_filter.get_heatmap(subsampling_factor)
    ind = np.random.choice(heatmap.size, size=1, p=heatmap.ravel())
    coord = np.unravel_index(ind, heatmap.shape)
    world_coord = np.array(coord) / subsampling_factor

    sensor_location = world_coord.reshape([1,2])

particle_filter.render(sensor_location, source_locations, render_heatmap=False)
plt.ioff()
plt.show()
