import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from Problems.Single_Objective import Single_Objective


#-----------------------------------------#
#-------------class Rastrigin-------------#
#-----------------------------------------#
class Rastrigin(Single_Objective):
    """Class for the single-objective Rastrigin problem.

    :param n_dvar: number of decision variable
    :type n_dvar: positive int, not zero
    """

    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#
    
    #-------------__init__-------------#    
    def __init__(self, n_dvar):
        Single_Objective.__init__(self, n_dvar)

    #-------------__del__-------------#
    def __del__(self):
        Single_Objective.__del__(self)
        
    #-------------__str__-------------#
    def __str__(self):
        return "Rastrigin problem "+str(self.n_dvar)+" decision variables "+str(self.n_obj)+" objective"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------perform_real_evaluation-------------#
    def perform_real_evaluation(self, candidates):
        """Objective function.

        :param candidates: candidate decision vectors
        :type candidates: np.ndarray
        :return: objective values
        :rtype: np.ndarray
        """

        assert self.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])
        obj_vals = 10*self.n_dvar + np.sum(candidates.__pow__(2)-10*np.cos(2*np.pi*candidates), axis=1)
        return obj_vals

    #-------------get_bounds-------------#
    def get_bounds(self):
        """Returns search space bounds.

        :returns: search space bounds
        :rtype: np.ndarray
        """

        res=np.ones((2,self.n_dvar))
        res[0,:]*=-4.12
        res[1,:]*=7.12
        return res

    #-------------is_feasible-------------#
    def is_feasible(self, candidates):
        """Check feasibility of candidates.

        :param candidates: candidate decision vectors
        :type candidates: np.ndarray
        :returns: boolean indicating whether candidates are feasible
        :rtype: bool
        """

        res=False
        if Single_Objective.is_feasible(self, candidates)==True:
            lower_bounds=self.get_bounds()[0,:]
            upper_bounds=self.get_bounds()[1,:]
            res=(lower_bounds<=candidates).all() and (candidates<=upper_bounds).all()
        return res

    #-------------plot-------------#
    def plot(self):
        """Plot the 1D or 2D Rastrigin objective function."""
        
        if self.n_dvar==1:
            x = np.linspace(self.get_bounds()[0], self.get_bounds()[1], 100)
            y = self.perform_real_evaluation(x)

            plt.plot(x, y, 'r')
            plt.title("Rastrigin-1D")
            plt.show()
            
        elif self.n_dvar==2:
            fig = plt.figure()

            lower_bounds = self.get_bounds()[0,:]
            upper_bounds = self.get_bounds()[1,:]

            x = np.linspace(lower_bounds[0], upper_bounds[0], 100)
            y = np.linspace(lower_bounds[1], upper_bounds[1], 100)
            z = self.perform_real_evaluation( np.array(np.meshgrid(x, y)).T.reshape(-1,2) ).reshape(x.size, y.size)
            x, y = np.meshgrid(x, y)
            
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet, antialiased=False)
            plt.title("Rastrigin-2D")
            plt.show()

        else:
            print("[Rastrigin.py] Impossible to plot Rastrigin with n_dvar="+str(self.n_dvar))
