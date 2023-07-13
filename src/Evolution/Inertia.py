import numpy as np

from Evolution.Velocities_Updater import Velocities_Updater
from Evolution.Swarm import Swarm


#---------------------------------------#
#-------------class Inertia-------------#
#---------------------------------------#
class Inertia(Velocities_Updater):
    """Class for an updater of velocities with intertia coefficient.

    `Frans Van den Bergh. An analysis of particle swarm optimizers [Ph. D. dissertation]. Department of Computer Science, University of Pretoria, Pretoria, South Africa (2002)`_

    `J. Kennedy and R. Eberhart, “Particle swarm optimization,” in Proc. IEEE Int. Conf. Neural Networks, 1995, pp. 1942–1948.`_

    :param inertia: inertia coefficient
    :type inertie: float
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, c1=2.0, c2=2.0, inertia=None):
        Velocities_Updater.__init__(self, c1, c2)
        assert type(inertia)==float or inertia is None
        if inertia is None or inertia<=0.5*(c1+c2)-1:
            inertia=0.5*(c1+c2)-1+0.1
            
        self.__inertia = inertia

    #-------------__del__-------------#
    def __del__(self):
        Velocities_Updater.__del__(self)
        del self.__inertia

    #-------------__str__-------------#
    def __str__(self):
        return "Velocities updater with inertia coefficient "+str(self.__inertia)+", first acceleration coefficient "+str(self._c1)+", second acceleration coefficient "+str(self._c2)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------perform_velocities_update-------------#    
    def perform_velocities_update(self, swarm, idx_best_nb):
        """Update the velocities of the swarm with inertia coefficient.

        :param swarm: swarm to update the velocities
        :type swarm: Swarm
        :param idx_best_nb: for each particle, index of the best neighbor
        :type idx_best_nb: np.array
        """
        Velocities_Updater.perform_velocities_update(self, swarm, idx_best_nb)

        r1=np.random.uniform()
        r2=np.random.uniform()

        swarm.velocities = self.__inertia*swarm.velocities + self._c1*r1*(swarm.pbest_dvec - swarm.dvec) + self._c2*r2*(swarm.dvec[idx_best_nb] - swarm.dvec)
