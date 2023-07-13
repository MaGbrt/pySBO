from typing import Optional
import torch
import math

import Global_Var
from Global_Var import *

from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from torch import Tensor


class custom_qEI(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        y_max: Tensor,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([Global_Var.af_nsamples]))
        self.sampler = sampler
        self.register_buffer("y_max", torch.as_tensor(y_max))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        print('Forward custom qEI')        
        batch_ei = self.model.EI(X, self.y_max).requires_grad_(True)
        print('returns ei:', batch_ei)
        return batch_ei
    
    
    



class qScalarizedUpperConfidenceBound(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        beta: Tensor,
        weights: Tensor,
        sampler: Optional[MCSampler] = None,
    ) -> None:
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=model)
        if sampler is None:
            sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))
        self.sampler = sampler
        self.register_buffer("beta", torch.as_tensor(beta))
        self.register_buffer("weights", torch.as_tensor(weights))

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate scalarized qUCB on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Upper Confidence Bound values at the
                given design points `X`.
        """
        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)  # n x b x q x o
        scalarized_samples = samples.matmul(self.weights)  # n x b x q
        mean = posterior.mean  # b x q x o
        scalarized_mean = mean.matmul(self.weights)  # b x q
        ucb_samples = (
            scalarized_mean
            + math.sqrt(self.beta * math.pi / 2)
            * (scalarized_samples - scalarized_mean).abs()
        )
        return ucb_samples.max(dim=-1)[0].mean(dim=0)

