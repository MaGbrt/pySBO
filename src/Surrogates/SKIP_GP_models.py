#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:43:11 2023

@author: maxime
"""
import torch
import numpy as np
from Surrogates.Models import Model
#### CREATE CUSTOM MODEL #####
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.gpytorch import GPyTorchModel
from gpytorch import settings
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.settings import cholesky_jitter
from botorch.fit import fit_gpytorch_model
from sklearn.metrics import mean_squared_error
from botorch.optim.fit import fit_gpytorch_torch
from scipy.stats import norm
from scipy.optimize import minimize

dtype = torch.double
    
#%% SKIP GP
from gpytorch.kernels import ProductStructureKernel, GridInterpolationKernel

class SKIP_GP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, likelihood):
        super(SKIP_GP, self).__init__(train_X, train_Y.squeeze(-1), likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = MaternKernel()
        self.covar_module = ProductStructureKernel(
            ScaleKernel(
                GridInterpolationKernel(self.base_covar_module, grid_size=100, num_dims=1)
            ), num_dims=train_X.shape[-1]
        )
        self.to(train_X)     # make sure we're on the right device/dtype
        self.to(train_Y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    
#%%
class SKIP_GP_model(Model):
    
    def __init__(self, train_X, train_Y):
        self._likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3)) #GaussianLikelihood()
        self._model = SKIP_GP(train_X, train_Y, self._likelihood)
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
        with settings.max_preconditioner_size(10), torch.no_grad():
            with settings.use_toeplitz(False), settings.max_root_decomposition_size(30), settings.fast_pred_var():
                preds = self._model(points.double())
        return preds.mean, preds.variance
        
    
    def get_std(self, points):
        return 0

    def fit(self):
        return 0
    

    def EI(self, X_query, y_max, xi=0.01):
        mu, sigma = self.predict(X_query)
        with np.errstate(divide='warn'):
            dif = mu - y_max - xi
            Z = dif / sigma
            ei = dif * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        return ei.flatten()

    def custom_fit(self, budget = 200):
        # Find optimal model hyperparameters
        self._model.train()
        self._likelihood.train()
        memory = 10000000000000000.;
        best_mse = 100000000000.
        mini_batch = 5
        tol = 1e-3;
        patience = 3;
        stuck = 0;
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01)
        
        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)
        
        with settings.use_toeplitz(True):
            for i in range(budget):
                # Zero backprop gradients
                optimizer.zero_grad()
                with settings.use_toeplitz(False), settings.max_root_decomposition_size(30):
                    # Get output from model
                    output = self._model(self._train_X)
                    # Calc loss and backprop derivatives
                    loss = -mll(output, self._train_Y.flatten())
                    loss.backward()
                    
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
                optimizer.step()
        
 
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