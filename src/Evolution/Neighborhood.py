import numpy as np
from abc import ABC
from abc import abstractmethod

from Evolution.Population import Population


#-----------------------------------------------------#
#-------------abstract class Neighborhood-------------#
#-----------------------------------------------------#
class Neighborhood(ABC):
    """Abstract class for neighborhood.

    :param neighbors_idx: index of the neighbors of each particle in the swarm.
    :type neighbors_idx: np.ndarray
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    @abstractmethod
    def __init__(self, pop, n_clusters):
        assert isinstance(pop, Population)
        assert type(n_clusters)==int
        assert n_clusters<=pop.dvec.shape[0] and n_clusters>=1
        
        self._neighbors_idx = []
        self._n_clusters = n_clusters
    
    #-------------__del__-------------#
    def __del__(self):
        del self._neighbors_idx
        del self._n_clusters

    #-------------__str__-------------#
    @abstractmethod
    def __str__(self):
        pass

    
    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_neighbors_idx-------------#
    def _get_neighbors_idx(self):
        return self._neighbors_idx

    #-------------_set_neighbors_idx-------------#
    def _set_neighbors_idx(self, new_neighbors_idx):
        self._neighbors_idx=new_neighbors_idx

    #-------------_get_n_clusters-------------#
    def _get_n_clusters(self):
        return self._n_clusters

    #-------------_set_n_clusters-------------#
    def _set_n_clusters(self, new_n_clusters):
        self._n_clusters=new_n_clusters


    #-------------property-------------#
    neighbors_idx=property(_get_neighbors_idx, _set_neighbors_idx, None)
    n_clusters=property(_get_n_clusters, _set_n_clusters, None)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------idx_best_nb_per_part-------------#
    def idx_best_nb_per_part(self, swarm):
        """Returns the index (in the swarm) of the best neighbor for each particle.

        :param swarm: swarm of particles
        :type swarm: Swarm
        :returns: array of indexes (size of the array equals the swarm size)
        :rtype: np.array
        """
        assert isinstance(swarm, Population)

        if swarm.obj_vals.ndim>1 and swarm.obj_vals.shape[1]>1:
            print("[Neighborhood] multi-objective is not supported")
            assert False

        res = np.array([], dtype=int)

        for nbh_idxs in self._neighbors_idx:
            nbh_idxs = np.array(nbh_idxs)
            min_idx_obj_vals = np.argmin(swarm.obj_vals[nbh_idxs])
            res = np.append(res, nbh_idxs[min_idx_obj_vals])
        
        return res
