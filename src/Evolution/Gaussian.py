import numpy as np

from Evolution.Mutation import Mutation
from Evolution.Population import Population


#----------------------------------------#
#-------------class Gaussian-------------#
#----------------------------------------#
class Gaussian(Mutation):
    """Class for gaussian mutation.

    :param prob: probability of mutation
    :type prob: float in [0,1]
    :param sigma: stdev of the gaussian
    :type sigma: positive int, not zero
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, prob, sigma):
        Mutation.__init__(self, prob)
        assert type(sigma)==float
        assert sigma>0
        self.__sigma=sigma

    #-------------__del__-------------#
    def __del__(self):
        Mutation.__del__(self)
        del self.__sigma

    #-------------__str__-------------#
    def __str__(self):
        return "Gaussian mutation probability "+str(self.prob)+" noise standard deviation "+str(self.__sigma)


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_sigma-------------#
    def _get_sigma(self):
        return self.__sigma

    #-------------property-------------#
    sigma=property(_get_sigma, None, None)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------perform_mutation-------------#    
    def perform_mutation(self, pop):
        """Mutates the individuals of a population.

        :param pop: population to mutate
        :type pop: Population
        :returns: the mutated population
        :rtype: Population
        """
        Mutation.perform_mutation(self, pop)
        bounds = pop.pb.get_bounds()
        assert bounds.shape[1]==pop.dvec.shape[1]

        children = Population(pop.pb)
        children.dvec = np.copy(pop.dvec)

        # Loop over the children population
        for child in children.dvec:

            # choosing the decision variables to mutate
            nb_dvar_to_mutate = np.random.binomial(child.size, self.prob)
            if self.prob>0.0 and nb_dvar_to_mutate==0:
                nb_dvar_to_mutate=1
            dvar_to_mutate = np.random.choice(np.arange(0, child.size, 1, dtype=int), nb_dvar_to_mutate, replace=False)
            
            # sampling the noise to add
            noise = np.random.normal(0.0, self.__sigma, dvar_to_mutate.size)
            child[dvar_to_mutate] += noise
            
            # ensure the boundaries are respected
            child[np.where(child<bounds[0])[0]]=bounds[0][np.where(child<bounds[0])[0]]
            child[np.where(child>bounds[1])[0]]=bounds[1][np.where(child>bounds[1])[0]]

        return children
