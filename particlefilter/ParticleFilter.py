import numpy as np
from scipy.spatial import distance
from scipy.stats import poisson
from scipy import linalg
from sklearn import mixture
import matplotlib.pyplot as plt
import matplotlib as mpl

class ParticleFilter():
    def __init__(self, min_particle_strength, max_particle_strength, num_sources, world_size=20, num_particles=500, fusion_range=3):
        """
        For reference, I used min_particle_strength = 0.5 min(source_strength),
        max_particle_strength = 1.5 max(source_strength)
        """
        if isinstance(world_size, int):
            self.world_size = np.array([world_size, world_size], dtype=int)
        else:
            self.world_size = np.array(world_size, dtype=int)
        self.num_particles = num_particles
        self.num_sources = num_sources
        self.fusion_range_sq = fusion_range**2

        self._spawn_particles(min_particle_strength, max_particle_strength)

    def _spawn_particles(self, min_particle_strength, max_particle_strength):
        """Generate particles."""
        # min_particle_strength = 0.5 * np.min(self.source_strengths)
        # max_particle_strength = 1.5 * np.max(self.source_strengths)
        self.particle_locations = self.world_size * np.random.rand(self.num_particles, 2)
        self.particle_strength = np.random.uniform(min_particle_strength, max_particle_strength, size=self.num_particles)
        self.particle_weights = np.ones(self.num_particles) / self.num_particles

        # boolean mask tracking which particles have been within fusion range
        self.seen_particles = np.zeros(self.num_particles, dtype=bool)

    def get_heatmap(self, subsampling_factor=2):
        """
        Get an array representing particle density in the world, weighted by particle weight.

        Inputs:
            subsampling_factor - How many times to divide each world unit.

        Output:
            heatmap - subsampling_factor * world_size float array
        """
        normalized_weights = self.particle_weights / np.sum(self.particle_weights)
        heatmap, xedges, yedges = np.histogram2d(
            self.particle_locations[:,0],
            self.particle_locations[:,1],
            weights=normalized_weights,
            bins=subsampling_factor*self.world_size,
            range=[[0, self.world_size[0]], [0, self.world_size[1]]])

        return heatmap

    def fit_gaussian(self, num_sources, use_only_seen_particles=True):
        """Fit a guassian mixture model to the data. Returns the means and covariances."""
        particles = self.particle_locations[self.seen_particles] if use_only_seen_particles else self.particle_locations
        gmm = mixture.GaussianMixture(n_components=num_sources, covariance_type='full').fit(particles)
        means = gmm.means_
        covariances = gmm.covariances_

        return means, covariances

    def step(self, reading, sensor_location):
        """
        Update the particle filter with a new sensor reading. Requires a known sensor location.

        Inputs:
            reading - integer radiation count
            sensor_location - [1 x 2] float array. Note this has to be two dimensional.
        """
        # Get readings from particles
        particle_distance_sq = distance.cdist(sensor_location, self.particle_locations, 'sqeuclidean')
        particle_expected_intensity = self.particle_strength / (1 + particle_distance_sq)
        particle_reading = np.round(particle_expected_intensity)

        # Find particles within fusion range
        fusion_inliers = (particle_distance_sq <= self.fusion_range_sq)
        if np.sum(fusion_inliers) == 0:
            return

        # this converts a boolean mask to 1D list of integer indices
        fusion_inliers_idx = np.ravel_multi_index(np.where(fusion_inliers), fusion_inliers.shape)
        self.seen_particles[fusion_inliers_idx] = True

        # Compute weight update, f_y and p_star as in
        #   'Robust Localization of an Arbitrary Distribution of Radioactive Sources for Aerial Inspection'
        f_y = poisson.pmf(np.floor(particle_expected_intensity[fusion_inliers]),
                          particle_expected_intensity[fusion_inliers])
        p_star = poisson.pmf(reading, particle_expected_intensity[fusion_inliers]) / f_y

        # update weights
        p_star = np.squeeze(p_star)
        fusion_weights = self.particle_weights[fusion_inliers_idx]
        fusion_weights *= p_star
        fusion_weights /= np.sum(fusion_weights)

        # Resample
        num_fusion_inliers = np.sum(fusion_inliers)
        resample_idx = np.random.choice(num_fusion_inliers,
                                        size=num_fusion_inliers,
                                        p=fusion_weights)

        # we sampled the inliers, so map those indices to the original data indices
        resample_idx = fusion_inliers_idx[resample_idx]

        self.particle_locations[fusion_inliers_idx,:] = self.particle_locations[resample_idx,:]
        self.particle_strength[fusion_inliers_idx] = self.particle_strength[resample_idx]
        self.particle_weights[fusion_inliers_idx] = self.particle_weights[resample_idx]
        self.particle_weights /= np.sum(self.particle_weights)

        # Add noise, parameters from Dhruv's code
        self.particle_locations[fusion_inliers_idx,:] += np.random.normal(0.0, 0.2, [num_fusion_inliers, 2])
        self.particle_locations[:,0] = np.clip(self.particle_locations[:,0], 0, self.world_size[0])
        self.particle_locations[:,1] = np.clip(self.particle_locations[:,1], 0, self.world_size[1])
        self.particle_strength[fusion_inliers_idx]  += np.random.normal(0.0, 0.1, num_fusion_inliers)
        self.particle_strength = np.maximum(0, self.particle_strength)

    def render(self, sensor_location=None, source_locations=None, render_heatmap=True):
        """Render the particle filter. Returns the figure object."""

        # Normalize weights
        m = np.max(self.particle_weights)
        scaled_weights = self.particle_weights / m

        # Setup figure
        plt.clf()
        ax = plt.gca()

        # Heat map
        if render_heatmap:
            subsampling_factor = 2
            heatmap, xedges, yedges = np.histogram2d(self.particle_locations[:,0], self.particle_locations[:,1],
                weights=scaled_weights, bins=subsampling_factor*self.world_size,
                range=[[0, self.world_size[0]], [0, self.world_size[1]]])
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            ax.imshow(heatmap.T, extent=extent, origin='lower')

        # Border
        ax.add_patch(plt.Rectangle((0,0), self.world_size[0], self.world_size[1], fill=False))
        ax.axis('equal')
        ax.axis([-1, self.world_size[0]+1, -1, self.world_size[1]+1])

        # Particles
        ax.scatter(self.particle_locations[:,0], self.particle_locations[:,1], s=scaled_weights, c='k')

        if source_locations is not None:
            ax.scatter(source_locations[:,0], source_locations[:,1], c='g', marker='x')

        if sensor_location is not None:
            ax.scatter(sensor_location[:,0], sensor_location[:,1], c='c', marker='o')

        if source_locations is not None and np.sum(self.seen_particles) > 2:
            # Fit GMM and render as ellipses
            num_sources = len(source_locations)
            means, covariances = self.fit_gaussian(num_sources, use_only_seen_particles=True)

            ax.scatter(means[:,0], means[:,1], c='c', marker='x')
            for i, (mean, covar) in enumerate(zip(means, covariances)):
                v, w = linalg.eigh(covar)
                v = 2. * np.sqrt(2.) * np.sqrt(v)
                u = w[0] / linalg.norm(w[0])

                if np.max(v) < 3:
                    color = 'c'
                else:
                    color = 'y'
                # Plot an ellipse to show the Gaussian component
                angle = np.arctan(u[1] / u[0])
                angle = 180. * angle / np.pi  # convert to degrees
                ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
                ell.set_clip_box(ax.bbox)
                ell.set_alpha(0.5)
                ax.add_artist(ell)

        return plt.gcf()
