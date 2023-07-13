import numpy as np
import math

from Evolution.Velocities_Updater import Velocities_Updater
from Evolution.Swarm import Swarm


#-----------------------------------------------#
#-------------class Dynamic_Inertia-------------#
#-----------------------------------------------#
class Dynamic_Inertia(Velocities_Updater):
    """Class for an updater of velocities with dynamic inertia coefficient.

    Y. Shi and R. C. Eberhart, Empirical study of particle swarm optimization, in Proc. IEEE Int. Congr. Evolutionary Computation, vol. 3, 1999, pp. 101–106.

    :param search_budget: budget for the search (either expressed in number of iterations or time)
    :type search_budget: positive int, not zero
    :param w_min: lower bound for the inertia coefficient
    :param w_min: float
    :param w_max: upper bound for the inertia coefficient
    :param w_max: float
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, search_budget, c1=2.0, c2=2.0, w_max=0.9, w_min=0.4):
        Velocities_Updater.__init__(self, c1, c2)
        assert search_budget>0
        assert w_max>w_min
        
        self._search_budget = search_budget
        self._w_max = w_max
        self._w_min = w_min

    #-------------__del__-------------#
    def __del__(self):
        Velocities_Updater.__del__(self)
        del self._search_budget
        del self._w_max
        del self._w_min

    #-------------__str__-------------#
    def __str__(self):
        return "Velocities updater with dynamic inertia coefficient, budget for the search "+str(self._search_budget)+", first acceleration coefficient "+str(self._c1)+", second acceleration coefficient "+str(self._c2)+", maximum value of the inertia coefficient "+str(self._w_max)+", minimum value of the inertia coefficient "+str(self._w_min)


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_search_budget-------------#
    def _get_search_budget(self):
        return self._search_budget

    #-------------_get_w_max-------------#
    def _get_w_max(self):
        return self._w_max

    #-------------_get_w_min-------------#
    def _get_w_min(self):
        return self._w_min

    #-------------property-------------#
    search_budget=property(_get_search_budget, None, None)
    w_max=property(_get_w_max, None, None)
    w_min=property(_get_w_min, None, None)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------perform_velocities_update-------------#    
    def perform_velocities_update(self, swarm, idx_best_nb, search_progress):
        """Return the updated velocities of the swarm with dynamic inertia coefficient.

        :param swarm: swarm to update the velocities
        :type swarm: Swarm
        :param idx_best_nb: for each particle, index of the best neighbor
        :type idx_best_nb: np.array
        :param search_progress: current search progress (expressed either in number of iterations or time)
        :type search_progress: positive int
        """
        Velocities_Updater.perform_velocities_update(self, swarm, idx_best_nb)

        r1=np.random.uniform()
        r2=np.random.uniform()
        inertia = self._w_max - (self._w_max-self._w_min)*search_progress/self._search_budget

        swarm.velocities = inertia*swarm.velocities + self._c1*r1*(swarm.pbest_dvec - swarm.dvec) + self._c2*r2*(swarm.dvec[idx_best_nb] - swarm.dvec)
