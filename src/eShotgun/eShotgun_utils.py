import scipy
import numpy as np
import torch
import os
from eShotgun.batch_methods.acquisition_optimisers import aquisition_CMAES, minimise_with_CMAES
from eShotgun.batch_methods.acquisition_optimisers import aquisition_LBFGSB, minimise_with_LBFGSB
from eShotgun.batch_methods.nsga2_pareto_front import NSGA2_pygmo
from time import time

def estimate_L(model, xj, lengthscale, lb, ub):
    """
    Estimate the Lipschitz constant of f by taking maximizing the
    norm of the expectation of the gradient of *f*.

    Adapated from GPyOpt:
        GPyOpt/core/evaluators/batch_local_penalization.py

    """
    def df(x, model):
        x = np.atleast_2d(x)
        # Get into evaluation (predictive posterior) mode
        model._model.eval()
        model._model.likelihood.eval()
        
        X = torch.autograd.Variable(torch.Tensor(x), requires_grad=True)
        observed_pred = model._model.likelihood(model._model(X))
        y = observed_pred.mean.sum()
        y.backward()
        dydx = X.grad.numpy()
#        print('Compute the gradient: ', dydx)
        # simply take the norm of the expectation of the gradient
        res = np.sqrt((dydx * dydx).sum(1))
        
        # bfgs (scipy 1.5.0) expects shape (d,) rather than (1,d)
        if x.shape[0] == 1:
            res = res[0]
            
        return -res

    # generate bounds, box constraint on xj
    n_dim = model._dim

    # centred on xj, lengthscale wide in each dimension, subject to the domain
    df_lb = np.maximum(xj - lengthscale, lb)
    df_ub = np.minimum(xj + lengthscale, ub)

    # scipy bounds
    bounds = list(zip(df_lb, df_ub))

    # generate some samples in the box around xj and evaluate their gradient
    samples = np.random.uniform(df_lb, df_ub, size=(500, n_dim))
    samples = np.vstack([samples, model._train_X])

    samples_df = df(samples, model)

    # select the starting point as that with the largest (negative) gradient
    x0 = samples[np.argmin(samples_df)]

    xopt, minusL, _ = scipy.optimize.fmin_l_bfgs_b(df,
                                                   x0,
                                                   bounds=bounds,
                                                   args=(model,),
                                                   maxiter=2000,
                                                   approx_grad=True)

    L = -np.squeeze(minusL).item()

    if L < 1e-7:
        L = 10  # to avoid problems in cases in which the model is flat.

    return L


def calculate_ball_radius(xj, model, lb, ub):
    # r_j \leq ||\mu(x_j) - M|| / L   +  \gamma * \sigma(x_j) / L
    # where L = estimated Lipschitz constant locally
    #           within a lengthscale of x_j
    # gamma = 1
    # M = min Ytr
    # calculates the radius of the ball to sample in, centred on xj
    ls = model._model.covar_module.base_kernel.lengthscale.squeeze().detach().numpy()
    # locally estimate the Lipshitz constant as the largest gradient within
    # a lengthscale of xj and within the problem domain
    L = estimate_L(model, xj, ls, lb, ub)
    # estimate the best function value to be the best seen function value
    M = np.min(model._train_Y.numpy())
    # gamma: Asynchronous Batch Bayesian Optimisation
    #        with Improved Local Penalisation
    gamma = 1
    p = model.make_preds(np.atleast_2d(xj))
    mu_xj = p.mean.detach().numpy()
    sigmaSQR_xj = p.variance.detach().numpy()
    sigma_xj = np.sqrt(sigmaSQR_xj)

    rj = (np.abs(mu_xj - M) + (gamma * sigma_xj)) / L

    return np.squeeze(rj)


def generate_batch(model, f_lb, f_ub, feval_budget, q,
                    epsilon, pf=False, cf = None, aq_func=lambda mu, sigma: -mu):
    n_dim = model._dim
    # epsilon of the time, randomly choose a point in space
    t_start = time()
    if np.random.uniform() < epsilon:
        # print('Trigger epsilon: calculate the pareto front and randomly select a location on it')
        if pf:
            # print('Trigger epsilon: calculate the pareto front and randomly select a location on it')
            # calculate the pareto front and randomly select a location on it
            X_front, musigma_front = NSGA2_pygmo(model, feval_budget,
                                                 f_lb, f_ub, cf)
            xj = X_front[np.random.choice(X_front.shape[0]), :]

        else:
            # print('Trigger epsilon: pick a random point')
            xj = np.random.uniform(f_lb, f_ub)
        t_epsilon = time()
        # print('time for triggering epsilon: ', t_epsilon - t_start)
    # else find the point that maximises an acquisition function
    else:
        # print('Find the point that maximises an acquisition function')
        # we can only use CMA-ES on 2 or more dimensional functions
        if n_dim > 1:
            opt_acq_func, opt_caller = aquisition_CMAES, minimise_with_CMAES

        # else use L-BFGS-B
        else:
            opt_acq_func, opt_caller = aquisition_LBFGSB, minimise_with_LBFGSB

        # cma-es wrapper for it (this just negates it)
        f = opt_acq_func(model, aq_func, cf)

        # minimise the negative acquisition function with cma-es
        xj = opt_caller(f, f_lb, f_ub, feval_budget, cf=cf)
        # print('Get the points: ', xj)
        t_epsilon = time()
        # print('time for optimizing the AF: ', t_epsilon - t_start)
    
    # radius of the ball around xj in which to sample new locations
    rj = calculate_ball_radius(xj, model, f_lb, f_ub)
    t_ball = time()
    # print('time for calculate ball radius: ', t_ball - t_epsilon)
    

    # maximum ball radius = half the size of norm of the domain
    rj = np.minimum(rj, np.linalg.norm(f_ub - f_lb) / 2)

    Xnew = [xj]

    # sample new locations, x_i ~ N(xj, rj), where N is the normal distribution
    # print('sample new locations, x_i ~ N(xj, rj), where N is the normal distribution')
    t_fill = time()
    while len(Xnew) < q:
        Xtest = np.random.normal(loc=xj, scale=rj)

        # ensuring batch location lies within the problem domain
        while True:
            # ndarray of indices of locations out of the domain
            bad_inds = np.flatnonzero(np.logical_or(Xtest < f_lb,
                                                    Xtest > f_ub))

            # if they're all in bounds, break the checking loop
            if bad_inds.size == 0:
                break

            # isotropic scaling so we can individually sample components
            if bad_inds.size > 0:
                Xtest[bad_inds] = np.random.normal(loc=xj[bad_inds], scale=rj)

        Xnew.append(Xtest)
    t_end = time()
    # print('Time to fill the batch: ', t_end - t_fill )
    # print('Total time for the AP: ', t_end - t_start)
    Xnew = np.array(Xnew)
    # print('Generate batch gives: ', Xnew)
    return torch.tensor(Xnew)


def egreedy_shotgun_v2(model, f_lb, f_ub, feval_budget, q, cf,
                       epsilon, pf=False, aq_func=lambda mu, sigma: -mu):

    n_dim = f_lb.size

    # always select the (estimated) best point in the acquisition function
    # we can only use CMA-ES on 2 or more dimensional functions
    if n_dim > 1:
        opt_acq_func, opt_caller = aquisition_CMAES, minimise_with_CMAES

    # else use L-BFGS-B
    else:
        opt_acq_func, opt_caller = aquisition_LBFGSB, minimise_with_LBFGSB

    # CMA-ES wrapper for it (this just negates it)
    f = opt_acq_func(model, aq_func, cf)

    # minimise the negative acquisition function with CMA-ES
    xj = opt_caller(f, f_lb, f_ub, feval_budget, cf=cf)

    Xnew = [xj]

    # decide how many random samples to place
    r = np.random.uniform(0, 1, size=q - 1)
    n_random = np.count_nonzero(r < epsilon)

    # generate the remaining points randomly
    if n_random > 0:
        if pf:
            # calculate the Pareto front
            X_front, _ = NSGA2_pygmo(model, feval_budget, f_lb, f_ub, cf)

            # check the front is large enough
            n_front_samples = np.minimum(n_random, X_front.shape[0])

            # if not then uniformly sample the extra points needed
            for _ in range(n_random - n_front_samples):
                Xnew.append(np.random.uniform(f_lb, f_ub))

            # randomly select locations on the front to evaluate
            inds = np.random.choice(X_front.shape[0], size=n_front_samples,
                                    replace=False)
            for ind in inds:
                Xnew.append(X_front[ind, :])

        else:
            for _ in range(n_random):
                Xnew.append(np.random.uniform(f_lb, f_ub))

    # sample the remaining points via the shotgun blast approach
    if len(Xnew) < q:
        # radius of the ball around xj in which to sample new locations
        rj = calculate_ball_radius(xj, model, f_lb, f_ub)

        # maximum ball radius = half the size of norm of the domain
        rj = np.minimum(rj, np.linalg.norm(f_lb - f_ub) / 2)

        # sample new locations, x_i ~ N(xj, rj), where N is the normal
        # distribution
        while len(Xnew) < q:
            Xtest = np.random.normal(loc=xj, scale=rj)

            # ensuring that they lie within the problem domain
            if np.logical_and(np.all(Xtest >= f_lb), np.all(Xtest <= f_ub)):
                Xnew.append(Xtest)

    Xnew = np.array(Xnew)

    return Xnew
