from numpy import ma
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from time import time

from Problems.Single_Objective import Single_Objective

from julia.api import Julia

#--------------------------------------#
#------------- class PHES -------------#
#--------------------------------------#

class PHES(Single_Objective):
    """Class for the single-objective PHES problem.

    :param dim: number of decision variable
    :type dim: positive int, >1
    """

    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#    
    def __init__(self, n_dvar):
        self._jl = Julia(compiled_modules=False)
        self._jl.eval('include("./Problems/PHES/evaluate.jl")')
        assert n_dvar==30, 'The PHES problem is only define for 30 decision variables'
        Single_Objective.__init__(self, n_dvar)

    #-------------__del__-------------#
    def __del__(self):
        Single_Objective.__del__(self)

    #-------------__str__-------------#
    def __str__(self):
        return "PHES problem "+str(self.n_dvar)+" decision variables "+str(self.n_obj)+" objective"


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
        start = time()
        assert self.is_feasible(candidates)
        DAM_d = np.zeros(24)
        reserves_d = np.zeros(24)
        
        if candidates.ndim==1:
            DAM_d = candidates[:24]
            reserves_d = candidates[24:]
            instruction = 'evaluate_d(' + str(DAM_d) + ',' + str(reserves_d) + ')'
            instruction = instruction.replace("\n", "")
            p = self._jl.eval(instruction)
            p = np.array([p])

        else:
            p = np.zeros(len(candidates))
            #print('I have ', len(candidates), ' candidates to evaluate')
            k = 0
            for cand in candidates:
                DAM_d = cand[:24]
                reserves_d = cand[24:]
                instruction = 'evaluate_d(' + str(DAM_d) + ', ' + str(reserves_d) + ')'
                instruction = instruction.replace("\n", "")
                p[k] = self._jl.eval(instruction)
                k += 1
                
        end = time()
#        print('Evaluation done, returns: ', p, ' in ', end-start, ' s.')
        return -p


    #-------------get_bounds-------------#
    def get_bounds(self):
        """Returns search space bounds.

        :returns: search space bounds
        :rtype: np.ndarray
        """
        b=np.ones((2,self.n_dvar))
        b[0,:24]*=-10
        b[0,24:]*=0
        b[1,:]*=10

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
        print("[PHES.py] Impossible to plot PHES")
