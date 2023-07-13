import numpy as np
import math

from Evolution.Velocities_Updater import Velocities_Updater
from Evolution.Swarm import Swarm


#--------------------------------------------#
#-------------class Constriction-------------#
#--------------------------------------------#
class Constriction(Velocities_Updater):
    """Class for an updater of velocities with constriction coefficient.

    `Khokhar, MA., Boudt, K., Wan, C. (2021). Cardinality-Constrained Higher-Order Moment Portfolios Using Particle Swarm Optimization. In: Applying Particle Swarm Optimization. International Series in Operations Research & Management Science, vol 306. Springer, Cham. <https://doi.org/10.1007/978-3-030-70281-6_10>`_

    `Clerc, M., & Kennedy, J. (2011). The particle swarm—Explosion, stability, and convergence in a multidimensional complex space. IEEE Transactions on Evolutionary Computation, 6(1), 2011.`_

    :param kappa: parameter for the constriction coefficient
    :type kappa: float
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, c1=2.05, c2=2.05, kappa=0.5):
        Velocities_Updater.__init__(self, c1, c2)
        assert type(kappa)==float and kappa>=0.0 and kappa<=1.0
        self.__kappa = 0.5

    #-------------__del__-------------#
    def __del__(self):
        Velocities_Updater.__del__(self)
        del self.__kappa

    #-------------__str__-------------#
    def __str__(self):
        return "Velocities updater with constriction coefficient, kappa="+str(self.__kappa)+", first acceleration coefficient "+str(self._c1)+", second acceleration coefficient "+str(self._c2)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------perform_velocities_update-------------#    
    def perform_velocities_update(self, swarm, idx_best_nb):
        """Update the velocities of the swarm with constriction coefficient.

        :param swarm: swarm to update the velocities
        :type swarm: Swarm
        :param idx_best_nb: for each particle, index of the best neighbor
        :type idx_best_nb: np.array
        """
        Velocities_Updater.perform_velocities_update(self, swarm, idx_best_nb)
        
        r1=np.random.uniform()
        r2=np.random.uniform()
        phi=self._c1*r1+self._c2*r2
        if phi>4:
            constrict=2*self.__kappa/abs(2-phi-math.sqrt(phi*(phi-4)))
        else:
            constrict=self.__kappa

        swarm.velocities = constrict*(swarm.velocities + self._c1*r1*(swarm.pbest_dvec - swarm.dvec) + self._c2*r2*(swarm.dvec[idx_best_nb] - swarm.dvec))
