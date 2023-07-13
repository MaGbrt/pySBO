from abc import ABC
from abc import abstractmethod
import numpy as np

from Evolution.Swarm import Swarm
from Evolution.Positions_Updater import Positions_Updater


#---------------------------------------------------------#
#-------------abstract class Reflecting_Walls-------------#
#---------------------------------------------------------#
class Reflecting_Walls(Positions_Updater):
    """Class for positions updater with reflecting walls."""
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self):
        Positions_Updater.__init__(self)
    
    #-------------__del__-------------#
    def __del__(self):
        Positions_Updater.__del__(self)

    #-------------__str__-------------#
    def __str__(self):
        return "Positions updater with reflecting walls"

    
    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------perform_positions_update-------------#
    def perform_positions_update(self, swarm):
        """Update the positions of the particles in the swarm.

        To ensure the feasibility of the new positions, the technique of 
        reflecting walls is used. If a decision variable gets out of the
        boundaries, it is set to the boundary value and the associated
        velocity component is multiplied by -1.

        :param swarm: swarm for which to update positions
        :type swarm: Swarm
        """

        Positions_Updater.perform_positions_update(self, swarm)

        swarm.dvec = swarm.dvec + swarm.velocities

        l_bounds, u_bounds = swarm.pb.get_bounds()
        # lower bound
        (idx_dim_1, idx_dim_2) = np.where(swarm.dvec-l_bounds<0)
        swarm.dvec[idx_dim_1, idx_dim_2]=l_bounds[idx_dim_2]
        swarm.velocities[idx_dim_1, idx_dim_2]=-swarm.velocities[idx_dim_1, idx_dim_2]
        # upper bound
        (idx_dim_1, idx_dim_2) = np.where(u_bounds-swarm.dvec<0)
        swarm.dvec[idx_dim_1, idx_dim_2]=u_bounds[idx_dim_2]
        swarm.velocities[idx_dim_1, idx_dim_2]=-swarm.velocities[idx_dim_1, idx_dim_2]
