#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 10:49:35 2022

@author: maxime
"""
import torch
import numpy as np
import Global_Var
from Global_Var import *
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
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.settings import cholesky_jitter
from botorch.fit import fit_gpytorch_model
from sklearn.metrics import mean_squared_error
from botorch.optim.fit import fit_gpytorch_torch

dtype = torch.double


class My_ExactGP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, likelihood):
        # squeeze output dim before passing train_Y to ExactGP
        super(My_ExactGP, self).__init__(train_X, train_Y.squeeze(-1), likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=MaternKernel(ard_num_dims=train_X.shape[-1], nu=2.5), lengthscale_constraint=Interval(0.005, 4.0))
        
        self.to(train_X)     # make sure we're on the right device/dtype
        self.to(train_Y)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class GP_model(Model):
    
    def __init__(self, train_X, train_Y):
        self._likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3)) #GaussianLikelihood()
        self._model = My_ExactGP(train_X, train_Y, self._likelihood)
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
        post = self._model(points.double())
        return self._likelihood(post).mean
        
    def make_preds(self, points):
        self._model.eval() 
        self._likelihood.eval()
        if (len(points.shape) == 1):
            points = points.unsqueeze(0)       
        f_preds = self._model(points.double())
        f_mean = f_preds.mean
        f_cov = f_preds.covariance_matrix
        print(f_preds, f_mean, f_cov)
        # Make predictions by feeding model through likelihood
        observed_pred = self._likelihood(self._model(points.double()))
        print(observed_pred)
        return observed_pred
    
    def get_std(self, points):
        return 0

    def fit(self):
        with cholesky_jitter(Global_Var.chol_jitter):
    #        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)
            mll = ExactMarginalLogLikelihood(self._model.likelihood, self._model)
       
    #        print('MSE before training for this model is ', self.evaluate(self._train_X, self._train_Y))
            self._model.train()
            self._likelihood.train()
            try :
                fit_gpytorch_model(mll)
                
            except :
                print('Exception, tying with another optimisation method to fit the model')
                print(self._train_X)
                print(self._train_Y)
                try :
                    fit_gpytorch_model(mll, optimizer=fit_gpytorch_torch) # Optimizer uses Adam
                except :
                    print('Error in fitting the model')
                    return 0
                
    #        print('MSE after training for this model is ', self.evaluate(self._train_X, self._train_Y))
    
    def custom_fit(self, budget=200):
        self._model.train()
        self._likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)
        mll = mll.to(self._train_X)
        memory = 10000000000000000.;
        best_mse = 100000000000.
        mini_batch = 10
        tol = 1e-3;
        patience = 3;
        stuck = 0;
        self._model.train()
        self._likelihood.train()
        for i in range(budget):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self._model(self._train_X)
            # Calc loss and backprop gradients
            loss = -mll(output, self._train_Y.flatten())
            loss.backward()
            optimizer.step()
            
            if (i%mini_batch)==0:
#                print('Iter %d/%d - Loss: %.3f ' % (i + 1, budget, loss.item() ))
                if(loss.item() < memory - tol):
                    memory = np.min((loss.item(), memory));
                    stuck = 0;
                else:
                    stuck += 1;
                    if (stuck >= patience):
 #                       print('Stuck, early stopping at ', i)
                        break;

            # if (i%mini_batch)==0:
    #             mse = mean_squared_error(self._train_Y.detach().numpy(), self._likelihood(self._model(self._train_X)).mean.detach().numpy() )
    #             best_mse = min(mse, best_mse)
    #             print('Iter %d/%d - Loss: %.3f ' % (i + 1, budget, loss.item() ))
    #             print('mse = ', mse, 'best_mse = ', best_mse)
                

    # #            mse = self.evaluate(self._train_X, self._train_Y)
    #             if(mse < memory - tol):
    #                 memory = np.min((mse, memory));
    #                 stuck = 0;
    #             else:
    #                 memory = mse
    #                 stuck += 1;
    #                 if (stuck >= patience):
    #                     print('Stuck, early stopping')
    #                     break;
                    
        return loss.detach().numpy()
            


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
       
    
    def test(self, n_points):
        self._model.eval()
        self._likelihood.eval()
        print('Calling test function')
        test_x = torch.rand(n_points,self._dim).double()
        print('testing \n', test_x)
        f_preds = self._model(test_x)
        print('Getting GP latent function \n', f_preds)
        y_preds = self._likelihood(self._model(test_x))
        print('Getting the posterior distribution \n', y_preds)
        f_mean = f_preds.mean
        print('mean ', f_mean)
        f_var = f_preds.variance
        print('variance ', f_var)
        
    
#%% SKIP GP
from gpytorch.kernels import ProductStructureKernel, GridInterpolationKernel

class SKIP_GP(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, likelihood):
        super(SKIP_GP, self).__init__(train_X, train_Y.squeeze(-1), likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = RBFKernel()
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
        return preds.mean
        
    
    def get_std(self, points):
        return 0

    def fit(self, budget = 200):
        # Find optimal model hyperparameters
        self._model.train()
        self._likelihood.train()
        
        # Use the adam optimizer
        optimizer = torch.optim.Adam(self._model.parameters(), lr=0.01)
        
        # "Loss" for GPs - the marginal log likelihood
        mll = ExactMarginalLogLikelihood(self._likelihood, self._model)
        
        for i in range(budget):
            # Zero backprop gradients
            optimizer.zero_grad()
            with settings.use_toeplitz(False), settings.max_root_decomposition_size(30):
                # Get output from model
                output = self._model(self._train_X)
                # Calc loss and backprop derivatives
                loss = -mll(output, self._train_Y.flatten())
                loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, budget, loss.item()))
            optimizer.step()

    def get_std(self, points):
        return 0

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