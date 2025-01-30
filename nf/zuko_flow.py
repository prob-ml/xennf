import torch
import zuko
from typing import Tuple

import pyro
import pyro.distributions as dist
from torch import Size, Tensor

from zuko.lazy import UnconditionalTransform

class ZukoToPyro(pyro.distributions.TorchDistribution):
    r"""Wraps a Zuko distribution as a Pyro distribution.

    If ``dist`` has an ``rsample_and_log_prob`` method, like Zuko's flows, it will be
    used when sampling instead of ``rsample``. The returned log density will be cached
    for later scoring.

    :param dist: A distribution instance.
    :type dist: torch.distributions.Distribution

    .. code-block:: python

        flow = zuko.flows.MAF(features=5)

        # flow() is a torch.distributions.Distribution

        dist = flow()
        x = dist.sample((2, 3))
        log_p = dist.log_prob(x)

        # ZukoToPyro(flow()) is a pyro.distributions.Distribution

        dist = ZukoToPyro(flow())
        x = dist((2, 3))
        log_p = dist.log_prob(x)

        with pyro.plate("data", 42):
            z = pyro.sample("z", dist)
    """

    def __init__(self, dist: torch.distributions.Distribution):
        self.dist = dist
        self.cache = {}

    @property
    def has_rsample(self) -> bool:
        return self.dist.has_rsample

    @property
    def event_shape(self) -> Size:
        return self.dist.event_shape

    @property
    def batch_shape(self) -> Size:
        return self.dist.batch_shape

    def __call__(self, shape: Size = ()) -> Tensor:
        if hasattr(self.dist, "rsample_and_log_prob"):  # fast sampling + scoring
            x, self.cache[x] = self.dist.rsample_and_log_prob(shape)
        elif self.has_rsample:
            x = self.dist.rsample(shape)
        else:
            x = self.dist.sample(shape)

        return x

    def log_prob(self, x: Tensor) -> Tensor:
        if x in self.cache:
            return self.cache[x]
        else:
            return self.dist.log_prob(x)

    def expand(self, *args, **kwargs):
        return ZukoToPyro(self.dist.expand(*args, **kwargs))
    
    def rsample(self, sample_shape=torch.Size()):
        return self.dist.rsample(sample_shape)  # Delegate to the underlying flow

    def sample(self, sample_shape=torch.Size()):
        return self.dist.sample(sample_shape)  # Delegate to the underlying flow
    
def retrieve_activation(activation_str):

    match activation_str:
        case "ReLU":
            return torch.nn.ReLU
        case "LeakyReLU":
            return torch.nn.LeakyReLU
        case "Tanh":
            return torch.nn.Tanh
        case "Sigmoid":
            return torch.nn.Sigmoid
        case "ELU":
            return torch.nn.ELU
        case "SELU":
            return torch.nn.SELU
        case _:
            return torch.nn.Tanh

def setup_zuko_flow(flow_type: str, num_clusters: int, flow_length: int = 1, context_length: int = 0, hidden_layers: Tuple[int, ...] = None, activation: str = "Tanh"):
    match flow_type:
        case "MAF":
            cluster_probs_flow_dist = zuko.flows.autoregressive.MAF(
                features=num_clusters,
                context=context_length,
                transforms=flow_length,
                hidden_features=hidden_layers,
                activation=retrieve_activation(activation),
                # residual=True
            )
        case "CNF":
            cluster_probs_flow_dist = zuko.flows.continuous.CNF(
                features=num_clusters,
                context=context_length,
                hidden_features=hidden_layers,
                activation=retrieve_activation(activation),
                # atol=1e-5, #TURN THESE UP FOR LOWER MEMORY/COMP COST, DOWN FOR BETTER ACCURACY
                # rtol=1e-4,
                exact=False,
                # normalize=True,
            )
        case "GF":
            cluster_probs_flow_dist = zuko.flows.gaussianization.GF(
                features=num_clusters,
                context=context_length,
                transforms=flow_length,
                components=num_clusters,
                hidden_features=hidden_layers,
                activation=retrieve_activation(activation),
                normalize=True
            )
        # case "MNN":
        #     cluster_probs_flow_dist = zuko.flows.neural.MNN(
        #         signal=16,
        #         hidden_features=hidden_layers,
        #         activation=retrieve_activation(activation),
        #         normalize=True
        #     )
        case "NAF":
            cluster_probs_flow_dist = zuko.flows.neural.NAF(
                features=num_clusters,
                context=context_length,
                transforms=flow_length,
                # hidden_features=hidden_layers,
                # activation=retrieve_activation(activation),
                signal=64,
                network={
                    "hidden_features": hidden_layers,
                    "activation": torch.nn.Tanh,
                    "normalize": True
                }
            )
        # case "UMNN":
        #     cluster_probs_flow_dist = zuko.flows.neural.UMNN(
        #         signal=16,
        #         hidden_features=hidden_layers,
        #         activation=retrieve_activation(activation),
        #         normalize=True
        #     )
        case "UNAF":
            cluster_probs_flow_dist = zuko.flows.neural.UNAF(
                features=num_clusters,
                context=context_length,
                transforms=flow_length,
                hidden_features=hidden_layers,
                activation=retrieve_activation(activation),
                signal=16
            )
        case "BPF":
            cluster_probs_flow_dist = zuko.flows.polynomial.BPF(
                features=num_clusters,
                context=context_length,
                degree=16,
                hidden_features=hidden_layers,
                activation=retrieve_activation(activation),
            )
        case "SOSPF":
            cluster_probs_flow_dist = zuko.flows.polynomial.SOSPF(
                features=num_clusters,
                context=context_length,
                degree=8,
                polynomials=4,
                hidden_features=hidden_layers,
                activation=retrieve_activation(activation),
            )
        case "NCSF":
            cluster_probs_flow_dist = zuko.flows.spline.NCSF(
                features=num_clusters,
                context=context_length,
                hidden_features=hidden_layers,
                activation=retrieve_activation(activation),
                bins=16
            )
        case "NSF":
            cluster_probs_flow_dist = zuko.flows.spline.NSF(
                features=num_clusters,
                context=context_length,
                hidden_features=hidden_layers,
                activation=retrieve_activation(activation),
                bins=16,
                # passes=2
            )

    # clamping
    # Manually wrap the flow with the SoftclipTransform
  
    return cluster_probs_flow_dist
