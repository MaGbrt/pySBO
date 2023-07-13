import numpy as np
from platypus import NSGAII, Problem, Real, Solution, InjectedPopulation, Archive
from smt.sampling_methods import LHS
from cvxopt import solvers, matrix
from itertools import combinations
from sklearn.cluster import KMeans
from Surrogates.GPyTorch_models import GP_model
import Global_Var
from Global_Var import *
import gpytorch
import torch
class ABAFMo:
    def __init__(self, DB, batch_size, lb=None, ub=None, alpha = 0.1, xita = 0.1, lamda = 1, mo_eval=200, M = 2, pop_size= 50):
        self._pop_size      = pop_size      # Population size for MOEA        
        self._alpha         = alpha         # Control parameter in mNSGA-II
        self._xita          = xita          # Control parameter in NHLA
        self._lamda         = lamda         # Control parameter in NHLA
        self._mo_eval       = mo_eval       # Total function evaluation number for one MOEA evaluation

        self._M             = M             # Number of acquisition functions
        self._batch_size    = batch_size    # Number of candidates provided by the acquisition process
        self._dim           = DB._dim       # Dimension of the problem
        
        if (lb == None and ub == None):
            self._lb            = np.zeros(self._dim)   # Lower bounds
            self._ub            = np.ones(self._dim)   # Upper bounds
        else:
            self._lb = lb
            self._ub = ub
        # self.dbx: the previously selected X excepted ones recommended in last iteration
        # self.dby: the previously evaluated results Y excepted ones recommended in last iteration
        # self.lsx: the recommended solutions in last iteration
        # self.lsy: the evaluation results of recommended solutions in last iteration
        # self.best_y: the best evaluation result so far
        n_first = DB._size - batch_size
        self._dbx = DB._X[:n_first, :].detach().numpy()
        self._dby = DB._y[:n_first].detach().numpy()
        self._lsx = DB._X[n_first:, :].detach().numpy()
        self._lsy = DB._y[n_first:].detach().numpy()
        self._best_y = DB._y.min().detach().numpy()
        self._last_best_y = np.min(self._dby)
        scaled_y = (self._dby - np.min(self._dby)) / (np.max(self._dby)-np.min(self._dby))

        self._model = GP_model(torch.tensor(self._dbx), torch.tensor(scaled_y))
        with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
            # Fit the model
            print('Fit model')
            self._model.custom_fit(Global_Var.medium_fit)
            
    def print_status(self):
        print('X_n-1: ', self._dbx)
        print('y_n-1: ', self._dby)
        print('B_n-1: ', self._lsx)
        print('B_n-1: ', self._lsy)
        print('Best y: ', self._best_y)
        print('Last best y: ', self._last_best_y)
        return None
        
    def NLHA(self):
        # Offline version of NLHA
        # @article{10.1162/evco_a_00223,
        #     author = {Li, Yifan and Liu, Hai-Lin and Goodman, E. D.},
        #     title = "{Hyperplane-Approximation-Based Method for Many-Objective Optimization Problems with Redundant Objectives}",
        #     journal = {Evolutionary Computation},
        #     volume = {27},
        #     number = {2},
        #     pages = {313-344},
        #     year = {2019},
        #     month = {06},
        #     doi = {10.1162/evco_a_00223},
        #     url = {https://doi.org/10.1162/evco\_a\_00223},
        # }
        # Magnitude adjustment of each objective
        F = np.zeros(np.shape(self.pf))
        for i in range(np.shape(F)[1]):
            F[:, i] = (self.pf[:, i] - np.min(self.pf[:, i])) / (np.max(self.pf[:, i]) - np.min(self.pf[:, i]))
        # Choose a proper constant $q$ for power transformation
        q_options = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
        min_error = np.inf
        w_optimal = []
        # Implement cvxopt to solve the constrained quadratic programming problem, the best choice of $q$ is with the smallest error
        for q_power in q_options:
            F_q = np.power(F,q_power)
            P = matrix(F_q.T @ F_q)
            q = matrix((-np.ones((1,np.shape(F)[0]))@F_q + self._lamda/2*np.ones((1,np.shape(F)[1]))).T)
            G = matrix(-np.eye(np.shape(F)[1]))
            h = matrix(np.zeros((np.shape(F)[1], 1)))
            solvers.options['show_progress'] = False
            tol = 1e-5
            solvers.options['feastol'] = tol
            sol = solvers.qp(P,q,G,h)
            w = sol['x']
            error = sol['primal objective']
            if error < min_error:
                w_optimal = w
                min_error = error
        w_optimal = np.array(w_optimal).ravel()
        max_w = np.max(w_optimal)
        obj = []
        # Identify the essential objectives
        for i in range(len(w_optimal)):
            if w_optimal[i] > self._xita * max_w:
                obj.append(i)
        min_correlation_coefficient = np.inf
        min_idx = -1
        # In case of reduction to only one objective, select another one with the smallest correlation coefficient
        if len(obj) == 1:
            for i in range(np.shape(F)[1]):
                if i == obj[0]:
                    continue
                correlation_coefficient = np.corrcoef(np.concatenate((np.reshape(self.pf[:, obj[0]], (1,-1)), np.reshape(self.pf[:, i], (1,-1))), 0))[0, 1]
                if correlation_coefficient < min_correlation_coefficient:
                    min_correlation_coefficient = correlation_coefficient
                    min_idx = i
            obj.append(min_idx)
        return obj





    def generate_batch(self, itr):
        def obj_mNSGA_II_8(x):
            # Modify each objective to transform NSGA-II into mNSGA-II
            ei_0, ei_1, ei_2, pi_0, pi_1, pi_2, gp_lcb, mes = self._model.MACE_8(np.array([x]))
            # acq_val = [-ei_0, -ei_1, -ei_2, -pi_0, -pi_1, -pi_2, gp_lcb, -mes]
            acq_val = [gp_lcb, -ei_0, -pi_0, -ei_1, -pi_1, -ei_2, -pi_2, -mes]
            mean_val = np.mean(acq_val)
            f = [(1-self._alpha)*fi + self._alpha*mean_val for fi in acq_val]
            return f

        def obj_NSGA_II_8(x):
            ei_0, ei_1, ei_2, pi_0, pi_1, pi_2, gp_lcb, mes = self._model.MACE_8(np.array([x]))
            # acq_val = [-ei_0, -ei_1, -ei_2, -pi_0, -pi_1, -pi_2, gp_lcb, -mes]
            acq_val = [gp_lcb, -ei_0, -pi_0, -ei_1, -pi_1, -ei_2, -pi_2, -mes]
            return acq_val

        def obj_essential_NSGA_II(x):
            return np.array(obj_NSGA_II_8(x))[self.essential_combination_idx].tolist()

        def obj_essential_mNSGA_II(x):
            return np.array(obj_mNSGA_II_8(x))[self.essential_combination_idx].tolist()


        def obj_comb(x):
            # During enumeration of the 2-acquisition function combination
            # print('Sending to evaluation: ', x)
            # print('type: ', type(x))
            return np.array(obj_NSGA_II_8(x))[self.current_combination_idx].tolist()

        def obj_best(x):
            # Use the chosen acquisition functions as objectives to recommend solutions in this iteration
            return np.array(obj_NSGA_II_8(x))[self.best_combination_idx].tolist()

        arch = Archive() # Stores solutions that belong to approximate Pareto Set among all evaluations of mNSGA-II
        problem = Problem(self._dim, 8)
        for i in range(self._dim):
            problem.types[i] = Real(self._lb[i], self._ub[i])
        problem.function = obj_mNSGA_II_8

        if itr > 0:
            # In each iteration, the initial population for mNSGA-II is partly inherited from previous Pareto Set
            num_init = max(min(int(0.6 * np.shape(self.ps)[0]), 10), 1)
            idxs = np.arange(np.shape(self.ps)[0])
            idxs = np.random.permutation(idxs)
            idxs = idxs[0: num_init]
            init_s = [Solution(problem) for _ in range(num_init)]
            for i in range(num_init):
                init_s[i].variables = [x for x in self.ps[idxs[i], :]]
            gen = InjectedPopulation(init_s)
            algorithm = NSGAII(problem, population=self._pop_size, generator=gen, archive=arch)
        else:
            # In the beginning, the population for mNSGA-II are totally random
            algorithm = NSGAII(problem, population=self._pop_size, archive=arch)


        algorithm.run(self._mo_eval)
        print('Algorithm.run 0 DONE')
        if len(algorithm.result) > self._batch_size:
            optimized = algorithm.result
        else:
            optimized = algorithm.population

        self.pf = np.array([s.objectives for s in optimized])
        self.ps = np.array([s.variables for s in optimized])
        self.pf_8d = self.pf.copy()
        self.ps_8d = self.ps.copy()

        u_ps, u_idx = np.unique(self.ps_8d, return_index= True, axis= 0)
        u_idx = u_idx.tolist()
        self.pf_8d = self.pf_8d[u_idx]
        self.ps_8d = self.ps_8d[u_idx]

        # Remove identical solutions in PS
        # self.unique()
        # print(f'Before unique, ps shape: {np.shape(self.ps)}')
        u_ps, u_idx = np.unique(self.ps, return_index= True, axis= 0)
        u_idx = u_idx.tolist()
        self.pf = self.pf[u_idx]
        self.ps = self.ps[u_idx]
        # print('Print Pareto front and Pareto set: ', self.pf, self.ps)
        # print(f'After unique, ps shape: {np.shape(self.ps)}')
        try:
            # Get the essential objectives of original 10-d MaOP
            self.essential_combination_idx = self.NLHA()
        except ValueError:
            self.essential_combination_idx = [0,1,2,3,4,5,6,7]
        # print(f'Essential objectives: {self.essential_combination_idx}')
        best_quality = -np.inf
        self.best_combination_idx = None
        # The initial population of mNSGA-II for the 2-d MOP is partly inherited from the previous 10-d PS
        num_inherit = max(min(int(0.6 * np.shape(self.ps_8d)[0]), 10), 1)
        idxs = np.arange(np.shape(self.ps_8d)[0])
        idxs = np.random.permutation(idxs)
        idxs = idxs[0: num_inherit]
        print('# Enumerate the 2-objective combination among essential objectives(acquisition functions) \n # Judge each combination by exploiting information of solutions recommended in last iteration')
        for combination in combinations(self.essential_combination_idx, self._M):
            print('Eval combination: ', combination)
            # Enumerate the 2-objective combination among essential objectives(acquisition functions)
            # Judge each combination by exploiting information of solutions recommended in last iteration
            quality = 0
            obj_idx = np.array(combination)
            self.current_combination_idx = obj_idx

            # Finetune the PS of 2-d MOP
            arch = Archive()
            problem = Problem(self._dim, self._M)
            for i in range(self._dim):
                problem.types[i] = Real(self._lb[i], self._ub[i])
            problem.function = obj_comb
            init_s = [Solution(problem) for _ in range(num_inherit)]
            for i in range(num_inherit):
                init_s[i].variables = [x for x in self.ps_8d[idxs[i], :]]
            gen = InjectedPopulation(init_s)
            algorithm = NSGAII(problem, population=self._pop_size, generator=gen, archive=arch)
            algorithm.run(self._mo_eval)
            print('Algorithm.run 1 DONE')

            if len(algorithm.result) > self._batch_size:
                optimized = algorithm.result
            else:
                optimized = algorithm.population

            pf_M_obj = np.array([s.objectives for s in optimized])
            ps_M_obj = np.array([s.variables for s in optimized])

            u_ps, u_idx = np.unique(ps_M_obj, return_index=True, axis=0)
            u_idx = u_idx.tolist()
            pf_M_obj = pf_M_obj[u_idx]
            ps_M_obj = ps_M_obj[u_idx]

            # Calculate the quality of each combination, choose the one with highest quality
            print('# Calculate the quality of each combination, choose the one with highest quality')
            # print('Eval: ', self._lsx)
            # print('type: ', type(self._lsx))
            for i in range(np.shape(self._lsy)[0]):
                # print('Eval: ', self._lsx[i])
                # print('type: ', type(self._lsx[i]))
                # print('type: ', self._lsx[i].shape)
                pf_lsx = np.array(self._model.MACE_8(self._lsx[i].reshape(1, self._dim)))[obj_idx]
                # print('pf_lsx: ', pf_lsx)
                dis_list = [(pf_M_obj[j] - pf_lsx)@(pf_M_obj[j] - pf_lsx).T for j in range(np.shape(pf_M_obj)[0])]
                dis = np.min(dis_list)
                quality += (self._last_best_y - self._lsy[i])/(dis + 1)
                # print('quality: ', quality)
            if quality > best_quality:
                best_quality = quality
                self.best_combination_idx = obj_idx

        # Update the dataset and train the new GP model used to recommend solutions in this iteration
        print('# Update the dataset and train the new GP model used to recommend solutions in this iteration')
        self._dbx = np.concatenate((self._dbx, self._lsx), axis= 0)
        self._dby = np.concatenate((self._dby, self._lsy), axis= 0)
        self._last_best_y = self._best_y
        
        scaled_y = (self._dby - np.min(self._dby)) / (np.max(self._dby)-np.min(self._dby))
        self._model = GP_model(torch.tensor(self._dbx), torch.tensor(scaled_y))

        with gpytorch.settings.max_cholesky_size(Global_Var.max_cholesky_size):
            print('Update the model')
            self._model.custom_fit(Global_Var.medium_fit)

        arch = Archive()
        problem = Problem(self._dim, self._M)
        for i in range(self._dim):
            problem.types[i] = Real(self._lb[i], self._ub[i])
        problem.function = obj_best
        # The initial population of mNSGA-II for the 2-d MOP is partly inherited from the previous chosen 2-d PS
        num_inherit = max(min(int(0.6 * np.shape(self.ps)[0]), 10), 1)
        idxs = np.arange(np.shape(self.ps)[0])
        idxs = np.random.permutation(idxs)
        idxs = idxs[0: num_inherit]
        init_s = [Solution(problem) for _ in range(num_inherit)]
        for i in range(num_inherit):
            init_s[i].variables = [x for x in self.ps[idxs[i], :]]
        gen = InjectedPopulation(init_s)
        algorithm = NSGAII(problem, population=self._pop_size, generator=gen, archive=arch)
        algorithm.run(self._mo_eval)
        print('Second algorithm.run DONE')
        if len(algorithm.result) > self._batch_size:
            optimized = algorithm.result
        else:
            optimized = algorithm.population

        self.pf = np.array([s.objectives for s in optimized])
        self.ps = np.array([s.variables for s in optimized])
        u_ps, u_idx = np.unique(self.ps, return_index= True, axis= 0)
        u_idx = u_idx.tolist()
        self.pf = self.pf[u_idx]
        self.ps = self.ps[u_idx]

        if np.shape(self.pf)[0] < self._batch_size:
            # In some cases, it turns out the size of pareto set given by optimizing the 2d-MOP is smaller than the batch size
            # For this situation, we use the essential objectives to form the MOP and get corresponding pareto set
            print(f'{self.func_name}, batch_size: {self.k}, id: {self.random_state}, iter: {iter}, fall into essential objectives because size of {np.shape(self.pf)}')
            arch = Archive()
            problem = Problem(self._dim, len(self.essential_combination_idx))
            for i in range(self._dim):
                problem.types[i] = Real(self._lb[i], self._ub[i])
            if len(self.essential_combination_idx) > 3:
                problem.function = obj_essential_mNSGA_II
            else:
                problem.function = obj_essential_NSGA_II
            # The initial population is partly inherited from 8-d PS
            num_inherit = max(min(int(0.6 * np.shape(self.ps_8d)[0]), 10), 1)
            idxs = np.arange(np.shape(self.ps_8d)[0])
            idxs = np.random.permutation(idxs)
            idxs = idxs[0: num_inherit]
            init_s = [Solution(problem) for _ in range(num_inherit)]
            for i in range(num_inherit):
                init_s[i].variables = [x for x in self.ps_8d[idxs[i], :]]
            gen = InjectedPopulation(init_s)
            algorithm = NSGAII(problem, population=self._pop_size, generator=gen, archive=arch)
            algorithm.run(self._mo_eval)
            print('Third algorithm.run DONE')

            if len(algorithm.result) > self._batch_size:
                optimized = algorithm.result
            else:
                optimized = algorithm.population

            self.pf = np.array([s.objectives for s in optimized])
            self.ps = np.array([s.variables for s in optimized])
            u_ps, u_idx = np.unique(self.ps, return_index=True, index=0)
            u_idx = u_idx.tolist()
            self.pf = self.pf[u_idx]
            self.ps = self.ps[u_idx]
            # In some cases, even essential objectives may fail to get enough choice in pareto set.
            # This situation occurs when the essential objectives are exactly the previously used 2-d MOP.
            # And we will use the pareto set of the 8d MaOP.
            if np.shape(self.ps)[0] < self._batch_size:
                self.pf = self.pf_8d
                self.ps = self.ps_8d
                print(f'{self.func_name}, batch_size: {self.k}, id: {self.random_state}, iter: {iter}, fall into 8d because size of essential pf: {np.shape(self.pf)}, size of 8d pf: {np.shape(self.pf_8d)}')
            else:
                print(f'{self.func_name}, batch_size: {self.k}, id: {self.random_state}, iter: {iter}, size of essential pf: {np.shape(self.pf)}, size of 8d pf: {np.shape(self.pf_8d)}')
        # Choose extreme solutions of the two objective
        solutions_idx = []
        for i in range(np.shape(self.pf)[1]):
            solutions_idx.append(np.argmin(self.pf[:, i]))

        solutions_idx = list(set(solutions_idx))
        if len(solutions_idx) > self._batch_size:
            solutions_idx = solutions_idx[:self._batch_size]
        self._lsx = self.ps[solutions_idx]
        num_cluster = self._batch_size - np.shape(self._lsx)[0]
        if num_cluster > 0:
            self.ps = np.delete(self.ps, solutions_idx, 0)
            self.pf = np.delete(self.pf, solutions_idx, 0)
            # Cluster the remaining solutions in PS, choose the solution in each cluster with the smallest posterior mean value
            cluster_idx = KMeans(n_clusters= num_cluster).fit_predict(self.pf)
            valid_cluster_idx = list(set(cluster_idx))
            if len(valid_cluster_idx) < num_cluster:
                cluster_idx = KMeans(n_clusters=num_cluster).fit_predict(self.ps)
            for i in range(num_cluster):
                cluster_x = self.ps[np.where(cluster_idx == i)]
                idx = np.argmin([self._model.predict(x).detach().numpy() for x in cluster_x])
                self._lsx = np.concatenate((self._lsx, np.reshape(cluster_x[idx], (1,-1))), axis= 0)

        return self._lsx

    def update_state(self, DB):
        self._lsy = []
        n_first = DB._size - self._batch_size
        self._dbx = DB._X[:n_first, :].detach().numpy()
        self._dby = DB._y[:n_first].detach().numpy()
        self._lsx = DB._X[n_first:, :].detach().numpy()
        self._lsy = DB._y[n_first:].detach().numpy()
        self._best_y = DB._y.min().detach().numpy()
        self._last_best_y = np.min(self._dby)

