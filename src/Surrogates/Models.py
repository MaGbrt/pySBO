from abc import ABC
from abc import abstractmethod



#--------------------------------------------------#
#-------------abstract class Surrogate-------------#
#--------------------------------------------------#
class Model(ABC):
    """Abstract class for surrogate models.
    """
    #-------------__init__-------------#
    @abstractmethod
    def __init__(self):
        pass
    
    #-------------__del__-------------#
    @abstractmethod
    def __del__(self):
        pass

    #-------------train-------------#
    @abstractmethod
    def fit(self):
        """Trains the surrogate model."""
        pass

    @abstractmethod
    def evaluate(self, test_X, test_y):
        """Tests the surrogate model."""
        pass

    #-------------perform_prediction-------------#
    @abstractmethod
    def predict(self, candidates):
        """Returns the predicted objective value of the candidates.

        :param candidates: candidates
        :type candidates: np.ndarray
        :returns: predicted objective values
        :rtype: np.ndarray
        """        
        pass
    
    @abstractmethod
    def get_std(self, candidates):
        """Returns the standard deviation of the candidates.
        :param candidates: candidates
        :type candidates: np.ndarray
        :returns: standard deviation
        :rtype: np.ndarray
        """        
        pass
