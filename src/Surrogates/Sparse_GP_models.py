#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 13:20:29 2023

@author: maxime
"""

import torch
import numpy as np
import Global_Var
import tqdm.notebook as tqdm
from Surrogates.Models import Model
#### CREATE CUSTOM MODEL #####
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gpytorch import GPyTorchModel
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel, InducingPointKernel
from gpytorch.settings import cholesky_jitter
from sklearn.metrics import mean_squared_error

dtype = torch.double
    
#%% Sparse GP

class Sparse_GP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API
    
    def __init__(self, train_x, train_y, likelihood):
        super(Sparse_GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(MaternKernel())
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=train_x[:100, :].clone(), likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
#%%
class Sparse_GP_model(Model):
    
    def __init__(self, train_X, train_Y):
        self._likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3)) #GaussianLikelihood()
        self._model = Sparse_GP(train_X, train_Y, self._likelihood)
        self._dim = train_X.shape[-1]
        self._train_X = train_X
        self._train_Y = train_Y
       
    
    def __del__(self):
        del self._likelihood
        del self._model
        
    def predict(self, points):
        if (len(points.shape) == 1):
            points = points.unsqueeze(0)
        self._model.eval() # a trained GP model in eval mode returns a MultivariateNormal containing the posterior mean and covariance
        self._likelihood.eval()
        with torch.no_grad():
            preds = self._model(points.double())
            preds1 = self._model.likelihood(self._model(points))
        
        print(preds.mean - preds1.mean)
        return preds.mean
        
    
    def get_std(self, points):
        return 0

    def fit(self):
        return 0
    
    def custom_fit(self, budget = 200):
        # Find optimal model hyperparameters
        self._model.train()
        self._likelihood.train()
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01)
        
        memory = 10000000000000000.;
        best_mse = 100000000000.
        mini_batch = 5
        tol = 1e-3;
        patience = 3;
        stuck = 0;
        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)
        
        iterator = tqdm.tqdm(range(budget), desc="Train")
        for i in iterator:
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = self._model(self._train_X)
            # Calc loss and backprop derivatives
            loss = -mll(output, self._train_Y.flatten())
            loss.backward()
            iterator.set_postfix(loss=loss.item())
            optimizer.step()        
        
            if (i%mini_batch)==0:
                # print('Iter %d/%d - Loss: %.3f ' % (i + 1, budget, loss.item() ))
                if(loss.item() < memory - tol):
                    memory = np.min((loss.item(), memory));
                    stuck = 0;
                else:
                    stuck += 1;
                    if (stuck >= patience):
                        print('Stuck, early stopping at ', i)
                        break;

 
    def evaluate(self, test_X, test_Y):
        print('Note: the present function assumes that test_Y is rescaled or normalized according to the training set (if different)')
        self._model.eval()
        self._likelihood.eval()
       
        y_preds = self._likelihood(self._model(test_X)).mean
        if(torch.isnan(self._train_Y).any() == True):
            print('Train y: ', self._train_Y)
            print('Predictions: ', y_preds.unsqueeze(-1))
            print('Real values: ', test_Y)
            print('Difference: ', torch.abs(y_preds.unsqueeze(-1) - test_Y))
        mse = mean_squared_error(test_Y.numpy(), y_preds.detach().numpy())
        #print('MSE for this set is: ', mse)
        return mse