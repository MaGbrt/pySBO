import itertools
import csv
import copy

from Problems.Problem import Problem
from Evolution.Population import Population
from Global_Var import *

#-------------------------------------#
#-------------class Swarm-------------#
#-------------------------------------#
class Swarm(Population):
    """Class for the swarm of a particle swarm optimization algorithm.

    :param pb: problem
    :type pb: Problem
    :param dvec: decision vectors of the particles
    :type dvec: np.ndarray
    :param obj_vals: objective value associated with each particle
    :type obj_vals: np.ndarray
    :param fitness_modes: evaluation mode associated with each particle: True for real evaluation and False for prediction (surrogate evaluation)
    :type fitness_modes: np.ndarray
    :param pbest_dvec: best decision vector found so far for each particle
    :type pbest_dvec: np.ndarray
    :param pbest_obj_vals: best objective value found so far for each particle
    :type pbest_obj_vals: np.ndarray
    :param velocities: velocity vectors of the particles
    :type velocities: np.ndarray
    """

    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, pb):
        Population.__init__(self, pb)

        # mono-objective
        if self.pb.n_obj==1:
            self.pbest_dvec=np.empty((0,self.pb.n_dvar))
            self.pbest_obj_vals=np.empty((0,))
            self.velocities=np.empty((0,self.pb.n_dvar))
        else:
            print("[Swarm.py] Swarm does not support multi-objective")
            assert False


    #-------------__del__-------------#
    def __del__(self):
        Population.__del__(self)
        del self.pbest_dvec
        del self.pbest_obj_vals
        del self.velocities

    #-------------__str__-------------#
    def __str__(self):
        return "Swarm\n  Problem:\n  "+str(self.pb)+"\n  Decision vectors:\n  "+str(self.dvec)+"\n  Objective values:\n  "+str(self.obj_vals)+"\n  Fitness modes:\n  "+str(self.fitness_modes)+"\n Best decision vector per particle:\n "+str(self.pbest_dvec)+"\n Best objective value per particle:\n "+str(self.pbest_obj_vals)+"\n Velocities:\n "+str(self.velocities)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#


    #-------------print_shapes-------------#
    def print_shapes(self):
        """Prints the shapes of the arrays `dvec`, `obj_vals`, `fitness_modes`, `pbest_dvec`, `pbest_obj_vals` and `velocities` forming the swarm."""
        Population.print_shapes(self)
        print(self.pbest_dvec.shape)
        print(self.pbest_obj_vals.shape)
        print(self.velocities.shape)

        
    #-------------check_integrity-------------#
    def check_integrity(self):
        """Checks arrays' shapes are consistent.

        :returns: True for arrays consistency and False otherwise
        :rtype: bool
        """
        pop_integrity = Population.check_integrity(self)
        nb_particles = self.dvec.shape[0]==self.pbest_dvec.shape[0]==self.pbest_obj_vals.shape[0]==self.velocities.shape[0]
        nb_dvar = self.pb.n_dvar==self.pbest_dvec.shape[1]==self.velocities.shape[1]
        if self.pb.n_obj==1:
            nb_obj = self.pb.n_obj==self.pbest_obj_vals.ndim
        else:
            print("[Swarm.py] Warning! Swarm does not support multi-objective")
            nb_obj = self.pb.n_obj==self.pbest_obj_vals.shape[1]


        return pop_integrity and nb_particles and nb_dvar and nb_obj

    
    #-------------append-------------#
    def append(self, swarm):
        """Appends particles to the current swarm.
        
        :param swarm: particles to be appended
        :type swarm: Swarm
        """

        Population.append(self, swarm)
        self.pbest_dvec = np.vstack( (self.pbest_dvec, swarm.pbest_dvec) )
        self.velocities = np.vstack( (self.velocities, swarm.velocities) )
        # mono-objective
        if self.obj_vals.ndim==1:
            self.pbest_obj_vals = np.concatenate( (self.pbest_obj_vals, swarm.pbest_obj_vals) )
        # multi-objective
        else:
            print("[Swarm.py] Warning! Swarm does not support multi-objective")
            self.obj_vals = np.vstack( (self.obj_vals, swarm.obj_vals) )
            self.fitness_modes = np.vstack( (self.fitness_modes, swarm.fitness_modes) )
            self.pbest_obj_vals = np.vstack( (self.pbest_obj_vals, swarm.pbest_obj_vals) )


    #-------------sort-------------#
    def sort(self):
        """Sorts the swarm according to ascending particles' current objective value (single-objective) or non-dominated and crowded distance sorting (multi-objective).

        :returns: permutation of indexes
        :rtype: np.ndarray
"""

        idx = Population.sort(self)
        self.pbest_dvec=self.pbest_dvec[idx]
        self.pbest_obj_vals=self.pbest_obj_vals[idx]
        self.velocities=self.velocities[idx]

        return idx


    #-------------split_in_batches-------------#
    def split_in_batches(self, n_batch):
        """Splits the swarm in batches.

        :param n_batch: number of batches
        :type n_batch: positive int, not zero
        :returns: list of batches
        :rtype: list(Swarm)
        """

        # before evaluation
        assert self.obj_vals.size==0 and self.fitness_modes.size==0

        batches = [Swarm(self.pb) for i in range(n_batch)]
        batches_dvec = np.split(self.dvec, n_batch)
        batches_pbest_dvec = np.split(self.pbest_dvec, n_batch)
        batches_velocities = np.split(self.velocities, n_batch)
        batches_pbest_obj_vals = np.split(self.pbest_obj_vals, n_batch)

        for (batch, batch_dvec, batch_pbest_dvec, batch_velocities, batch_pbest_obj_vals) in zip(batches, batches_dvec, batches_pbest_dvec, batches_velocities, batches_pbest_obj_vals):
            batch.dvec = batch_dvec
            batch.pbest_dvec = batch_pbest_dvec
            batch.velocities = batch_velocities
            batch.pbest_obj_vals = batch_pbest_obj_vals

        return batches


    #-------------update_best_sim-------------#
    def update_best_sim(self, f_best_profile, T_list, f_hypervolume=None):
        """Updates the best decision vector and objective value both globally and per particle.

        For mono-objective:
        The best evaluated decision vector (minimisation assumed) is saved in `Global_Var.dvec_min` 
        The best objective value is saved in `Global_Var.obj_val_min`.
        Both best decision vector and its objective value are printed to a file.
        self.pbest_dvec and self.pbest_obj_vals are also updated

        For multi-objective:
        Not supported for the moment

        :param f_best_profile: filename for logging
        :type f_best_profile: str
        :param f_hypervolume: filename for logging hypervolume
        :type f_hypervolume: str
        """

        Population.update_best_sim(self, f_best_profile, T_list, f_hypervolume)

        # mono-objective
        if self.obj_vals.ndim==1: 
            assert self.dvec.shape[0]==self.obj_vals.size==self.fitness_modes.size==self.pbest_dvec.shape[0]==self.pbest_obj_vals.size

            # Update the best decision vector and objective value per particle
            for i in np.where(self.fitness_modes==True)[0]:
                if self.obj_vals[i]<self.pbest_obj_vals[i]:
                    self.pbest_obj_vals[i]=self.obj_vals[i]
                    self.pbest_dvec[i]=self.dvec[i]

        # multi-objective
        else:
            print("[Swarm.py] Swarm does not support multi-objective")
            assert False


    #-------------save_to_csv_file-------------#
    def save_to_csv_file(self, f_pop_archive, f_pop_annexe=None):
        """Prints the swarm to two CSV files.

        The 1st CSV file (f_pop_archive) is organized as follows:
        First row: number of decision variables, number of objectives, number of fitness modes
        Second row: lower bounds of the decision variables
        Third row: upper bounds of the decision variables
        Remaining rows (one per particle): decision variables, objective values, fitness modes

        The 2nd CSV file (f_pop_annexe) is organized as follow:
        First row: number of decision variables, number of objectives, number of decision variables (i.e. length of velocity vector)
        Second row: lower bounds of the decision variables
        Third row: upper bounds of the decision variables
        Remaining rows (one per particle): best known decision vector, best known objective values, velocity

        :param f_pop_archive: filename of the 1st CSV file.
        :type f_pop_archive: str
        :param f_pop_annexe: filename of the 2nd CSV file.
        :type f_pop_annexe: str
        """

        assert type(f_pop_annexe)==str or f_pop_annexe is None
        assert self.check_integrity()

        Population.save_to_csv_file(self, f_pop_archive)

        if f_pop_annexe is not None:
        
            with open(f_pop_annexe, 'w') as my_file:
                # writing number of decision variables, number of objectives and number of decision variables
                my_file.write(str(self.dvec.shape[1])+" "+str(self.obj_vals.shape[1] if len(self.obj_vals.shape)>1 else 1 if self.obj_vals.shape[0]>0 else 0)+" "+str(self.dvec.shape[1])+"\n")
                # writing bounds
                my_file.write(" ".join(map(str, self.pb.get_bounds()[0]))+"\n")
                my_file.write(" ".join(map(str, self.pb.get_bounds()[1]))+"\n")
                # writing each particles'
                if self.pbest_obj_vals.ndim==1: # mono-objective
                    for (pbest_dvec, pbest_obj_val, vel) in itertools.zip_longest(self.pbest_dvec, self.pbest_obj_vals, self.velocities, fillvalue=''):
                        my_file.write(" ".join(map(str, pbest_dvec))+" "+str(pbest_obj_val)+" "+" ".join(map(str, vel))+"\n")
                else: # multi-objective
                    print("[Swarm.py] Swarm does not support multi-objective")
                    assert False


    #-------------load_from_csv_file-------------#
    def load_from_csv_file(self, f_pop_archive, f_pop_annexe):
        """Loads the swarm from a couple of CSV files.

        The 1st CSV file (f_pop_archive) has to be organized as follows:
        First row: number of decision variables, number of objectives, number of fitness modes
        Second row: lower bounds of the decision variables
        Third row: upper bounds of the decision variables
        Remaining rows (one per particle): decision variables, objective values, fitness mode

        The 2nd CSV file (f_pop_annexe) has to be organized as follow:
        First row: number of decision variables, number of objectives, number of decision variables (i.e. length of velocity vector)
        Second row: lower bounds of the decision variables
        Third row: upper bounds of the decision variables
        Remaining rows (one per particle): best known decision vector, best known objective values, velocity
        
        :param f_pop_archive: filename of the CSV file
        :type f_pop_archive: str
        :param f_pop_annexe: filename of the 2nd CSV file.
        :type f_pop_annexe: str
        """

        Population.load_from_csv_file(self, f_pop_archive)
        
        assert type(f_pop_annexe)==str

        with open(f_pop_annexe, 'r') as my_file:
            # Counting the number of lines.
            reader = csv.reader(my_file, delimiter=' ')
            n_samples = sum(1 for line in reader) - 3
            my_file.seek(0)
        
            # First line: number of decision variables, number of objectives and number of velocity components (= number of decision variables)
            line = next(reader)
            n_dvar = int(line[0])
            n_obj = int(line[1])
            assert n_dvar==int(line[2])

            # Second line: lower bounds
            lower_bounds = np.zeros((n_dvar,))
            lower_bounds[0:n_dvar] = np.asarray(next(reader))
            assert lower_bounds.all()==self.pb.get_bounds()[0].all()

            # Third line: upper bounds
            upper_bounds = np.zeros((n_dvar,))
            upper_bounds[0:n_dvar] = np.asarray(next(reader))
            assert upper_bounds.all()==self.pb.get_bounds()[1].all()

            # Following lines contain (pbest_dvec, pbest_obj_vals, velocities)
            self.pbest_dvec = np.zeros((n_samples, n_dvar))
            self.pbest_obj_vals = np.zeros((n_samples,n_obj))
            self.velocities = np.zeros((n_samples, n_dvar))
            for i, line in enumerate(reader):
                self.pbest_dvec[i] = np.asarray(line[0:n_dvar])
                self.pbest_obj_vals[i,0:n_obj] = np.asarray(line[n_dvar:n_dvar+n_obj])
                self.velocities[i] = np.asarray(line[n_dvar+n_obj:])
            if self.pbest_obj_vals.shape[1]<2:
                self.pbest_obj_vals = np.ndarray.flatten(self.pbest_obj_vals)
                
        assert self.check_integrity()
