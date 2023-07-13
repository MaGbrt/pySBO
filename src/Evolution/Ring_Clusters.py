import numpy as np

from Evolution.Neighborhood import Neighborhood
from Evolution.Population import Population


#---------------------------------------------#
#-------------class Ring_Clusters-------------#
#---------------------------------------------#
class Ring_Clusters(Neighborhood):
    """Class for a cluster-based Ring neighborhood.

    Within a cluster, all the particles are connected to each other.
    The clusters are connected to each other according to the Ring pattern.
    The inter-cluster connection particles are choosen randomly.

    If the number of particles in the swarm equals the number of clusters, 
    this corresponds to the particle-based ring neighborhood.

    :param neighbors_idx: index of the neighbors of each particle in the swarm.
    :type neighbors_idx: np.ndarray
    :param n_clusters: number of clusters
    :type n_clusters: int
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, pop, n_clusters):
        """
        __init__ method's input

        :param pop: Population of candidate solutions
        :type pop: Population
        :param n_clusters: number of clusters
        :type n_clusters: int
        """

        Neighborhood.__init__(self, pop, n_clusters)

         # number of particles per cluster
        nb_part_per_clust = (pop.dvec.shape[0]//self._n_clusters)*np.ones((self._n_clusters,), dtype=int)
        for i in range(pop.dvec.shape[0]%self._n_clusters):
            nb_part_per_clust[i]+=1

        # set the intra-cluster neighbors for each particle
        for k, clust_size in enumerate(nb_part_per_clust):
            for i in range(clust_size):
                self._neighbors_idx.append([j+np.sum(nb_part_per_clust[:k]) for j in range(clust_size)])
                self._neighbors_idx[-1].remove(len(self._neighbors_idx)-1)

        # set the inter-cluster connections
        inter_clust_connect = np.array([], dtype=int)
        if np.min(nb_part_per_clust)>=self._n_clusters:
            replace_mode = False
        else:
            replace_mode = True

        # for each cluster, we choose the particles that will be connected to the other clusters
        for k, clust_size in enumerate(nb_part_per_clust):
            # choose randomly without replacement, 2 integers
            # in [[sum(nb_part_per_clust[:k]); sum(nb_part_per_clust[:k])+clust_size[[
            # and append them to inter_clust_connect
            inter_clust_connect = np.append(inter_clust_connect, np.random.choice(clust_size, 2, replace_mode)+np.sum(nb_part_per_clust[:k]))
                
        # set the inter-cluster neighbors
        for k in range(0,inter_clust_connect.size,2):
            self._neighbors_idx[inter_clust_connect[k]].append(inter_clust_connect[k-1])
            self._neighbors_idx[inter_clust_connect[k-1]].append(inter_clust_connect[k])


    #-------------__str__-------------#
    def __str__(self):
        return "Cluster-based Ring neighborhood"


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_neighbors_idx-------------#
    def _get_neighbors_idx(self):
        return self._neighbors_idx

    #-------------_get_n_clusters-------------#
    def _get_n_clusters(self):
        return self._n_clusters

    #-------------property-------------#
    neighbors_idx=property(_get_neighbors_idx, None, None)
    n_clusters=property(_get_n_clusters, None, None)
