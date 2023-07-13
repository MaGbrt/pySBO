from abc import ABC
from abc import abstractmethod

from Evolution.Swarm import Swarm


#-----------------------------------------------------------#
#-------------abstract class Velocities_Updater-------------#
#-----------------------------------------------------------#
class Velocities_Updater(ABC):
    """Abstract class for velocities updater.

    :param c1: first acceleration coefficient
    :type c1: float
    :param c2: second acceleration coefficient
    :type c2: float
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    @abstractmethod
    def __init__(self, c1, c2):
        assert type(c1)==float and type(c2)==float
        self._c1=c1
        self._c2=c2
        pass
    
    #-------------__del__-------------#
    @abstractmethod
    def __del__(self):
        del self._c1
        del self._c2
        pass

    #-------------__str__-------------#
    @abstractmethod
    def __str__(self):
        pass

    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_c1-------------#
    def _get_c1(self):
        return self._c1

    #-------------_set_c1-------------#
    def _set_c1(self, new_c1):
        self._c1=new_c1

    #-------------_get_c2-------------#
    def _get_c2(self):
        return self._c2

    #-------------_set_c2-------------#
    def _set_c2(self, new_c2):
        self._c2=new_c2

    #-------------property-------------#
    c1=property(_get_c1, _set_c1, None)
    c2=property(_get_c2, _set_c2, None)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------perform_velocities_update-------------#
    @abstractmethod
    def perform_velocities_update(self, swarm, idx_best_nb):
        assert isinstance(swarm, Swarm)
        pass
