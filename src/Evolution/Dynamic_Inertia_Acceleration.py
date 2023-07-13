import numpy as np
import math

from Evolution.Dynamic_Inertia import Dynamic_Inertia
from Evolution.Velocities_Updater import Velocities_Updater
from Evolution.Swarm import Swarm


#------------------------------------------------------------#
#-------------class Dynamic_Inertia_Acceleration-------------#
#------------------------------------------------------------#
class Dynamic_Inertia_Acceleration(Dynamic_Inertia):
    """Class for an updater of velocities with dynamic inertia coefficient and dynamic acceleration coefficients.

    `Ratnaweera, A., Halgamuge, S., & Watson, H. Self-organizing hierarchical particle swarm optimizer with time-varying acceleration coefﬁcients. 2004 IEEE Transactions on Evolutionary Computation, 8(3), 240–255.`_

    :param search_budget: budget for the search (either expressed in number of iterations or time)
    :type search_budget: positive int, not zero
    :param c1_init: initial value for the cognitive acceleration coefficient
    :type c1_init: float
    :param c1_fin: final value for the cognitive acceleration coefficient
    :type c1_fin: float
    :param c2_init: initial value for the social acceleration coefficient
    :type c2_init: float
    :param c2_fin: final value for the social acceleration coefficient
    :type c2_fin: float
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, search_budget, c1_init=2.5, c1_fin=0.5, c2_init=0.5, c2_fin=2.5):
        Dynamic_Inertia.__init__(self, search_budget, c1_init, c2_init)
        assert c1_init>c1_fin and c2_init<c2_fin
        
        self.__c1_init=c1_init
        self.__c1_fin=c1_fin
        self.__c2_init=c2_init
        self.__c2_fin=c2_fin

    #-------------__del__-------------#
    def __del__(self):
        Dynamic_Inertia.__del__(self)
        del self.__c1_init
        del self.__c1_fin
        del self.__c2_init
        del self.__c2_fin

    #-------------__str__-------------#
    def __str__(self):
        return Dynamic_Inertia.__str__(self)+", initial value for the cognitive acceleration coefficient "+str(self.__c1_init)+", final value for the cognitive acceleration coefficient "+str(self.__c1_fin)+", initial for the social acceleration coefficient "+str(self.__c2_init)+", final value for the social acceleration coefficient "+str(self.__c2_fin)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------perform_velocities_update-------------#    
    def perform_velocities_update(self, swarm, idx_best_nb, search_progress):
        """Update the velocities of the swarm with dynamic inertia coefficient.

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
        self._c1 = self.__c1_init + (self.__c1_fin-self.__c1_init)*search_progress/self._search_budget
        self._c2 = self.__c2_init + (self.__c2_fin-self.__c2_init)*search_progress/self._search_budget

        swarm.velocities = inertia*swarm.velocities + self._c1*r1*(swarm.pbest_dvec - swarm.dvec) + self._c2*r2*(swarm.dvec[idx_best_nb] - swarm.dvec)
