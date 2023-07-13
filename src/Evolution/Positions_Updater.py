from abc import ABC
from abc import abstractmethod

from Evolution.Swarm import Swarm


#----------------------------------------------------------#
#-------------abstract class Positions_Updater-------------#
#----------------------------------------------------------#
class Positions_Updater(ABC):
    """Abstract class for positions updater.
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    @abstractmethod
    def __init__(self):
        pass
    
    #-------------__del__-------------#
    @abstractmethod
    def __del__(self):
        pass

    #-------------__str__-------------#
    @abstractmethod
    def __str__(self):
        pass

    
    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------perform_positions_update-------------#
    @abstractmethod
    def perform_positions_update(self, swarm):
        assert isinstance(swarm, Swarm)
        pass
