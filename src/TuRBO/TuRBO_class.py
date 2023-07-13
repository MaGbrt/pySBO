#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 17:23:09 2022

@author: maxime
"""
import math
import torch
import warnings
import Global_Var
import numpy as np
from Global_Var import *
dtype = torch.double

from gpytorch.settings import cholesky_jitter
from dataclasses import dataclass
from botorch.acquisition import qExpectedImprovement
from botorch.acquisition import ExpectedImprovement
from botorch.generation import MaxPosteriorSampling
from botorch.optim import optimize_acqf
from torch.quasirandom import SobolEngine


@dataclass
class TurboState:
    dim: int
    batch_size: int
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


class TuRBO(TurboState):
    def __init__(self, dim, batch_size):
        self._state = TurboState(dim = dim, batch_size=batch_size)

    def update_state(self, Y_next):
        if max(Y_next) > self._state.best_value + 1e-3 * math.fabs(self._state.best_value):
            self._state.success_counter += 1
            self._state.failure_counter = 0
        else:
            self._state.success_counter = 0
            self._state.failure_counter += 1
    
        if self._state.success_counter == self._state.success_tolerance:  # Expand trust region
            self._state.length = min(2.0 * self._state.length, self._state.length_max)
            self._state.success_counter = 0
        elif self._state.failure_counter == self._state.failure_tolerance:  # Shrink trust region
            self._state.length /= 2.0
            self._state.failure_counter = 0
        self._state.best_value = max(self._state.best_value, max(Y_next).item())
        if self._state.length < self._state.length_min:
            self._state.restart_triggered = True
    
    def state_to_vec(self):
        state = np.zeros(11)
        state[0] = self._state.dim
        state[1] = self._state.batch_size
        state[2] = self._state.length
        state[3] = self._state.length_min
        state[4] = self._state.length_max
        state[5] = self._state.failure_counter
        state[6] = self._state.failure_tolerance
        state[7] = self._state.success_counter
        state[8] = self._state.success_tolerance
        state[9] = self._state.best_value
        state[10] = self._state.restart_triggered

        return state

    def vec_to_state(self, state):
        self._state.dim = int(state[0])
        self._state.batch_size = int(state[1])
        self._state.length = float(state[2])
        self._state.length_min = float(state[3])
        self._state.length_max = float(state[4])
        self._state.failure_counter = int(state[5])
        self._state.failure_tolerance = int(state[6])
        self._state.success_counter = int(state[7])
        self._state.success_tolerance = int(state[8])
        self._state.best_value = float(state[9])
        self._state.restart_triggered = bool(state[10])
        #print('State of turbo: ', state)
        return None

    def generate_batch(self,
        mod,  # GP model
        batch_size,
        n_candidates=None,  # Number of candidates for Thompson sampling
        num_restarts=10,
        raw_samples=512,
        acqf="ei",  # "ei" or "ts"
    ):
        X = mod._train_X
        Y = mod._train_Y
        model = mod._model
        assert acqf in ("ts", "ei", "ei2", "KB_ei")
        assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
        if n_candidates is None:
            n_candidates = min(5000, max(2000, 200 * X.shape[-1]))
        
        # Scale the TR to be proportional to the lengthscales
        x_center = X[Y.argmax(), :].clone()
        weights = model.covar_module.base_kernel.lengthscale.squeeze().detach()
        weights = weights / weights.mean()
        weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
        tr_lb = torch.clamp(x_center - weights * self._state.length / 2.0, 0.0, 1.0)
        tr_ub = torch.clamp(x_center + weights * self._state.length / 2.0, 0.0, 1.0)
    
        if acqf == "ts":
            dim = X.shape[-1]
            sobol = SobolEngine(dim, scramble=True)
            pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
            pert = tr_lb + (tr_ub - tr_lb) * pert
    
            # Create a perturbation mask
            prob_perturb = min(20.0 / dim, 1.0)
            mask = (
                torch.rand(n_candidates, dim, dtype=dtype, device=device)
                <= prob_perturb
            )
            ind = torch.where(mask.sum(dim=1) == 0)[0]
            mask[ind, torch.randint(0, dim - 1, size=(len(ind),), device=device)] = 1
    
            # Create candidate points from the perturbations and the mask        
            X_cand = x_center.expand(n_candidates, dim).clone()
            X_cand[mask] = pert[mask]
    
            # Sample on the candidate points
            thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():  # We don't need gradients when using TS
                X_next = thompson_sampling(X_cand, num_samples=batch_size)
    
        elif acqf == "ei":
            with cholesky_jitter(Global_Var.chol_jitter):
                ei = qExpectedImprovement(model, Y.max())
                X_next, acq_value = optimize_acqf(
                    ei,
                    bounds=torch.stack([tr_lb, tr_ub]),
                    q=batch_size,
                    num_restarts=Global_Var.af_nrestarts, 
                    raw_samples=Global_Var.af_nsamples, 
                    options=Global_Var.af_options
                    )
                
        elif acqf == "KB_ei":
            print('Remains to be implemented')    
        return X_next
