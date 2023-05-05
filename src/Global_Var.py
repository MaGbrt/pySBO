import numpy as np

class Global_Var:
    dvec_min=np.empty((0,))
    ref_point=np.empty((0,))
    obj_val_min=float("inf")
    best_hv=0
    test=0

    # Gaussian Processes related hyper-parameters
    max_cholesky_size = float("inf")  # Always use Cholesky
    chol_jitter = 1e-3

    # Budget allocated to the surrogate model fitting
    # (using GPytorch Models and custom fit function)
    large_fit = 500
    small_fit = 50
    medium_fit = 200
    
    # Parameters of the acquisition strategies
    af_nrestarts = 10
    af_nsamples = 512
    af_options = {}
    af_options['maxfun']=500
    af_options['iprint']=-1 # neg means no output
    af_options['method']='L-BFGS-B'
