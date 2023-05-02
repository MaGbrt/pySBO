import numpy as np
from numpy import ma
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from Problems.Single_Objective import Single_Objective
from Problems.CEC2014 import CEC2014

class Composition_AS(Single_Objective):
    """Class for the single-objective Composition_AS problem.

    :param dim: number of decision variable
    :type dim: positive int, >1
    """

    #-------------__init__-------------#    
    def __init__(self, n_dvar):
        Single_Objective.__init__(self, n_dvar)
        self._f2 = CEC2014(6, n_dvar)
        
    #-------------__del__-------------#
    def __del__(self):
        Single_Objective.__del__(self)

    #-------------__str__-------------#
    def __str__(self):
        return "Composition_AS problem "+str(self.n_dvar)+" decision variables "+str(self.n_obj)+" objective"

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
#       CEC2014 id 6 : Weierstrass function        
        c_wei = candidates
        obj_vals2 = self._f2.perform_real_evaluation(c_wei)
        assert self.is_feasible(candidates)
        
        if candidates.ndim==1:
            candidates = np.array([candidates])

        
        M_p = np.load('../src/Problems/M2_p.npy')# matrice de permutation
        shift = 13.23
        k = 0
        for cand in candidates:
            candidates[k] = M_p @ cand + shift
            k +=1
        c_alp = (candidates + 100)/20 # Scale from [-100; 100]^D into [0, 10]^D, default for Alpine2

        obj_vals1 = np.prod(np.sqrt(c_alp)*np.sin(c_alp), axis=1) # Alpine
        obj_vals = obj_vals1 * 0.
        
        for k in range(len(candidates)):
            cand = candidates[k]
            ss1 = np.sqrt(np.dot(cand-shift, cand-shift)) + 1
            ss2 = np.sqrt(np.dot(cand, cand)) + 1
            w1 = np.exp(-ss1/(2*self.n_dvar*100))/ss1
            w2 = np.exp(-ss2/(2*self.n_dvar*225))/ss2
            v1 = w1 / (w1+w2)
            v2 = w2 / (w1+w2)
            obj_vals[k] = v1 * 0.7 * obj_vals1[k] + v2 * 8*obj_vals2[k]
        
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
        self._bounds = b
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
        """Plot the 1D or 2-D Composition_AS objective function."""
        
        if self.n_dvar==1:
            x = np.linspace(self.get_bounds()[0], self.get_bounds()[1], 100)
            y = self.perform_real_evaluation(x)

            plt.plot(x, y, 'r')
            plt.title("Composition_AS-1D")
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
            plt.title("Composition_AS-2D")
            plt.show()

        else:
            print("[Composition_AS.py] Impossible to plot Composition_AS with n_dvar="+str(self.n_dvar))
