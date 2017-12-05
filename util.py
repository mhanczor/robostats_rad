import numpy as np
from scipy.spatial import distance

def greedy_nearest_neighbor(sources, targets):
    """
    Uniquely pair each source feature with a target feature. Use greedy nearest neighbor to compute pairs.
    Assumes both are M x N: a set of M features of dimension N
    """
    source_output = np.empty_like(sources)
    target_output = np.empty_like(targets)

    N = len(sources)
    distance_sq = distance.cdist(sources, targets, 'sqeuclidean')
    for i in range(N):
        min_idx = np.argmin(distance_sq)
        s,t = np.unravel_index(min_idx, distance_sq.shape)

        source_output[i,:] = sources[s,:]
        target_output[i,:] = targets[t,:]

        # Set these to inf to prevent them from being the minimum
        distance_sq[s,:] = np.inf
        distance_sq[:,t] = np.inf

    return source_output, target_output

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    a = np.random.uniform(0, 10, size=[10,2])
    b = np.random.uniform(0, 10, size=[10,2])

    a_nn, b_nn = greedy_nearest_neighbor(a,b)
    print(a_nn)

    plt.scatter(a[:,0], a[:,1], c='r')
    plt.scatter(b[:,0], b[:,1], c='g')

    N = len(a)
    for i in range(N):
        v1 = (a_nn[i,0], b_nn[i,0])
        v2 = (a_nn[i,1], b_nn[i,1])
        plt.plot(v1, v2, c='k')

    plt.show()
