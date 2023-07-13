import torch
import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import qmc
import random

# Data sets of this class are normalize or mapped into the [0,1] hyper-cube.
class DataBase(): 
    def __init__(self, f, size):
        self._size = size
        self._dim = f.n_dvar
        self._X = torch.zeros(size, f.n_dvar);
        self._y = torch.zeros(size, 1);
        self._obj = f
    
    def __del__(self):
        del self._size
        del self._dim
        del self._X
        del self._y
        del self._obj

    def copy(self):
        DB = DataBase(self._obj, self._size)
        DB._X = self._X.clone().detach()
        DB._y = self._y.clone().detach().flatten()
        if(torch.isnan(self._X).any() == True):
            print('DB copy X ', self._X)
        if(torch.isnan(self._y).any() == True):
            print('DB copy y ', self._y)

        return DB
    
    def copy_from_center(self, n_max, center):
        l_size = min(n_max, self._size)
        
        DB = DataBase(self._obj, l_size)
        DB._X = torch.zeros(l_size, self._dim, dtype=torch.float64)
        DB._y = torch.zeros(l_size, 1, dtype=torch.float64)
        distance = torch.cdist(self._X.type(torch.DoubleTensor), center.reshape(1,self._dim).type(torch.DoubleTensor), p = 1)
        order = torch.argsort(distance, dim=0)

        for i in range(l_size):
            k = int(order[i].clone().detach())
            DB._X[i] = self._X[k].clone().detach()
            DB._y[i] = self._y[k].clone().detach()
                        
        if(torch.isnan(self._X).any() == True):
            print('DB copy from center X ', self._X)
        if(torch.isnan(self._y).any() == True):
            print('DB copy from center y ', self._y)

        return DB

    def copy_from_best(self, n_max, best):
        l_size = min(n_max, self._size)
        
        DB = DataBase(self._obj, l_size)
        DB._X = torch.zeros(l_size, self._dim, dtype=torch.float64)
        DB._y = torch.zeros(l_size, 1, dtype=torch.float64)
        distance = torch.cdist(self._X.type(torch.DoubleTensor), best.reshape(1,self._dim).type(torch.DoubleTensor), p = 1)
        order = torch.argsort(distance, dim=0)

        for i in range(l_size):
            k = int(order[i].clone().detach())
            DB._X[i] = self._X[k].clone().detach()
            DB._y[i] = self._y[k].clone().detach()
                        
        if(torch.isnan(self._X).any() == True):
            print('DB copy from best X ', self._X)
        if(torch.isnan(self._y).any() == True):
            print('DB copy from best y ', self._y)

        return DB
             
    def eval_f(self, x): # In BO, the DataSet is mapped into [O,1]^n_dvar, before evaluation the candidate point must be unmapped
        bounds = self._obj.get_bounds()
        x_ = (x * (bounds[1] - bounds[0]) + bounds[0])
        y_ = self._obj.perform_real_evaluation(x_)
        
        return y_

    def create(self, seed = torch.rand(1)):
        torch.manual_seed(seed)
        self._X = torch.rand(self._size, self._dim, dtype=float)
        self._y = torch.from_numpy(self.eval_f(self._X.numpy()))
        
        if(torch.isnan(self._X).any() == True):
            print('DB create X ', self._X)
        if(torch.isnan(self._y).any() == True):
            print('DB create y ', self._y)
        print('Try distance in creation')
        self.try_distance()
        
    def par_create(self, comm, seed = torch.rand(1)):
        my_rank = comm.Get_rank()
        n_proc = comm.Get_size()

        if my_rank == 0:
            torch.manual_seed(seed)
            candidates = torch.rand(self._size, self._dim, dtype=float)
            self._X = None

            ## send to workers
            n_cand = np.zeros(n_proc, dtype = 'i')
            for c in range(len(candidates)):
                send_to = c%n_proc
                n_cand[send_to] += 1
            
            ## Broadcast n_cand
            comm.Bcast(n_cand, root = 0)
            for c in range(len(candidates)):
                send_cand = candidates[c].numpy()
                send_to = c%n_proc
                if (send_to != 0):
                    comm.send(send_cand, dest = send_to, tag = c)
            
            ## Evaluate
            for c in range(int(n_cand[0])):
                y_new = self.eval_f(candidates[n_proc*c].numpy())
                if self._X == None :
                    self._X = candidates[n_proc*c].unsqueeze(0)
                    self._y = torch.tensor(y_new)
                else :
                    self.add(candidates[n_proc*c].unsqueeze(0), torch.tensor(y_new))
            
            ## Gather
            for c in range(len(candidates)):
                get_from = c%n_proc
                if (get_from != 0):
                    recv_eval = comm.recv(source = get_from, tag = c)
                    self.add(candidates[c].unsqueeze(0), torch.tensor(recv_eval))

            if(torch.isnan(self._X).any() == True):
                print('DB create X ', self._X)
            if(torch.isnan(self._y).any() == True):
                print('DB create y ', self._y)
#            print('Try distance in creation')
            self.try_distance()
            
        else :
            n_cand = np.zeros(n_proc, dtype = 'i')
            comm.Bcast(n_cand, root = 0)
            cand = []
            for c in range(n_cand[my_rank]):
                cand.append(comm.recv(source = 0, tag = my_rank + c * n_proc))

            ## Evaluate
            y_new = []
            for c in range(n_cand[my_rank]):
                y_new.append(self.eval_f(cand[c]))
            ## Send it back
            for c in range(n_cand[my_rank]):
                comm.send(y_new[c], dest = 0, tag = my_rank + c * n_proc)
          

    def create_lhs(self, seed = torch.rand(1)):
        torch.manual_seed(seed)
        sampler = qmc.LatinHypercube(d=self._dim)
        sample = sampler.random(n=self._size)

        self._X = torch.tensor(sample)
        self._y = torch.from_numpy(self.eval_f(self._X.numpy())).unsqueeze(-1)
        if(torch.isnan(self._X).any() == True):
            print('DB create X ', self._X)
        if(torch.isnan(self._y).any() == True):
            print('DB create y ', self._y)
        print('Try distance in creation')
        self.try_distance()
        
    def try_distance(self, tol = 0.001):
        d_mat = pdist(self._X, 'chebyshev')
        if (min(d_mat) < tol):
            print('min chebyshev distance ', sorted(d_mat)[:10])
    
    def set_Xy(self, X, y):
        assert(X.shape[1] == self._dim)
        assert(len(y) == X.shape[0])

        del self._X
        del self._y
        if (isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor)):
            self._X == X.clone().detach()
            self._y == y.clone().detach()
        elif (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            self._X = torch.tensor(X)
            self._y = torch.tensor(y)
        else :
            raise TypeError('Check type of objects')
        self._size = len(self._X)
        
    def add(self, x, y):
        # print('DB add y ', y)
        # print('DB add X ', x)

        assert(isinstance(x, torch.Tensor))
        assert(isinstance(y, torch.Tensor))
        if len(x.shape)==1:
            x=x.reshape(1, self._dim)

        self._X = torch.cat([self._X, x])
        self._y = torch.cat([self._y, y])
        self._size = len(self._X)
#        print('Length of X and y: ', len(self._X), ' ', len(self._y))        
        if(torch.isnan(self._X).any() == True):
            print('DB add X ', self._X)
        if(torch.isnan(self._y).any() == True):
            print('DB add y ', self._y)
            print('DB add X ', self._X)
            
        assert(torch.isnan(self._y).any() == False)

        #print(self._size)
    def get_best(self):
        id_best = torch.argmin(self._y)
        y_best = self._y[id_best]
        x_best = self._X[id_best]
        return x_best, y_best

    def select_clean(self, sort_X, batch_size, n_leaves, tol):
        arg_min = torch.argmin(self._y)

        index = []
        k = 0
        k_ok = 0

        while (k_ok < batch_size and k < n_leaves):
            if torch.is_tensor(sort_X[k]):
                cand_k = sort_X[k].numpy().ravel()
            else :
                cand_k = sort_X[k]
            cand_ok = False
            for doe in self._X:
                if max(abs(doe-cand_k))>tol:
                    cand_ok = True
                else :
                    cand_ok = False
                    break
            if cand_ok == True:
                k_ok = k_ok + 1
                index.append(int (k))
            k = k + 1
            
        selected_candidates = list(np.array(sort_X)[index])

        k = len(index)
        if (k < batch_size):
            print('Need to create ', batch_size - k, ' new candidates')
        while k < batch_size:
            k = k+1
            new_cand = torch.max(torch.zeros(self._dim), torch.min(torch.ones(self._dim), self._X[arg_min] + (torch.rand(self._dim)-0.5)/20))
            selected_candidates.append(np.array(new_cand))
        return selected_candidates

    def read_txt(self, name):
        data = np.loadtxt(name, dtype='float', delimiter='\t')
        self._X = torch.tensor(data[:,:self._dim])
        self._y = torch.tensor(data[:,self._dim])
        self.try_distance()
        
    def save_txt(self, name):
        save_X = self._X.numpy()
        save_y = self._y.unsqueeze(1).numpy()
        save = np.concatenate((save_X, save_y), axis = 1)
        np.savetxt(name, save, fmt='%.8e', delimiter='\t', newline='\n', header='')

    def save_as_population(self, name):
        assert type(name)==str
        lb = self._obj.get_bounds()[0]
        ub = self._obj.get_bounds()[1]
        x_ = self._X.detach().numpy() * (ub - lb) + lb
        y_ = self._y.detach().numpy()
        with open(name, 'w') as my_file:
            my_file.write(str(self._dim) + " 1 1 \n")
            my_file.write(" ".join(map(str, lb)) + "\n")
            my_file.write(" ".join(map(str, ub)) + "\n")            
            for (x, y) in zip(x_, y_):
                my_file.write(" ".join(map(str, x)) + " " + str(y) + " " + str(1) + "\n")

    def min_max_y_scaling(self, Y_test = None):
        if Y_test == None:
            out = (self._y - self._y.min())/(self._y.max() - self._y.min())
        else:
            out = (Y_test - self._y.min())/(self._y.max() - self._y.min())
        return out
    
    def normal_y_scaling(self, Y_test = None):
        if Y_test == None:
            out = (self._y - self._y.mean())/self._y.std()
        else:
            out = (Y_test - self._y.mean())/self._y.std()            
        return out
    
    def reverse_min_max_y_scaling(self, y):
        out = y*(self._y.max() - self._y.min()) + self._y.min()
        return out
    
    def reverse_normal_y_scaling(self, y):
        out = y * self._y.std() + self._y.mean()
        return out
    
    def print_min(self):
        bounds = self._obj.get_bounds()
        i_min = torch.argmin(self._y)
        x = self._X[i_min]
        min_x = (x * (bounds[1] - bounds[0]) + bounds[0])
        print('Minimum found value is ', self._y[i_min], ' realized at point ', min_x)
        
    def get_min(self):
        i_min = torch.argmin(self._y)
        x_min = self._X[i_min]
        y_min = self._y[i_min]
        return x_min, y_min

    def create_subset(self, n_pts):
        sub_DB = self.copy()
        c = random.sample(range(sub_DB._size), n_pts)
        sub_DB._X = sub_DB._X[c]
        sub_DB._y = sub_DB._y[c]
        sub_DB._size = len(sub_DB._y)
        return sub_DB
        