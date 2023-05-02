from numpy import ma
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from Problems.Single_Objective import Single_Objective

#--------------------------------------#
#-------------class Hybrid_AW-------------#
#--------------------------------------#

class Hybrid_AW(Single_Objective):
    """Class for the single-objective Hybrid_AW problem.

    :param dim: number of decision variable
    :type dim: positive int, >1
    """

    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#    
    def __init__(self, n_dvar):
        assert n_dvar==10, 'The Hybrid_AW is only define for 10 decision variables'
        Single_Objective.__init__(self, n_dvar)

    #-------------__del__-------------#
    def __del__(self):
        Single_Objective.__del__(self)

    #-------------__str__-------------#
    def __str__(self):
        return "Hybrid_AW problem "+str(self.n_dvar)+" decision variables "+str(self.n_obj)+" objective"


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

        M_p = np.load('../src/Problems/M_p.npy')# matrice de permutation
        k = 0
        for cand in candidates:
            candidates[k] = M_p @ cand + 3.45
            k+=1
            
        alpine_mask = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 1], bool)
        schwefel_mask = ~alpine_mask    

        a = np.arange(len(alpine_mask))
        rm_alp = a[~alpine_mask]
        rm_sch = a[~schwefel_mask]
        
        c_alp = np.delete(candidates, rm_alp, 1)
        c_alp = (c_alp + 100)/20 # Scale into [0, 10]^D, default for Alpine2, instead of [-100, 100] for CEC2015

        c_sch = np.delete(candidates, rm_sch, 1)

        obj_vals1 = np.prod(np.sqrt(c_alp)*np.sin(c_alp), axis=1)        
        obj_vals2 = 418.9828872724338*len(rm_alp)-np.einsum('ij,ij->i', c_sch, np.sin(np.sqrt(np.abs(c_sch))))

        
        obj_vals = obj_vals1 + obj_vals2
        return obj_vals


    #-------------get_bounds-------------#
    def get_bounds(self):
        """Returns search space bounds.

        :returns: search space bounds
        :rtype: np.ndarray
        """
        b=np.ones((2,self.n_dvar))
        b[0,:]*=-100
        b[1,:]*=100
        return b

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
        """Plot the 1D or 2-D Hybrid_AW objective function."""
        
        if self.n_dvar==1:
            x = np.linspace(self.get_bounds()[0], self.get_bounds()[1], 100)
            y = self.perform_real_evaluation(x)

            plt.plot(x, y, 'r')
            plt.title("Hybrid_AW-1D")
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
            plt.title("Hybrid_AW-2D")
            plt.show()

        else:
            print("[Hybrid_AW.py] Impossible to plot Hybrid_AW with n_dvar="+str(self.n_dvar))
