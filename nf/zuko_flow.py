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

def setup_zuko_flow(flow_type: str, num_clusters: int, flow_length: int = 1, context_length: int = 0, hidden_layers: Tuple[int, ...] = None):
    match flow_type:
        case "MAF":
            cluster_probs_flow_dist = zuko.flows.MAF(
                features=num_clusters,
                context=context_length,
                transforms=flow_length,
                hidden_features=hidden_layers,
            )
        case "CNF":
            cluster_probs_flow_dist = zuko.flows.CNF(
                features=num_clusters,
                context=context_length,
            )
        case "GF":
            cluster_probs_flow_dist = zuko.flows.GF(
                features=num_clusters,
                context=context_length,
                transforms=flow_length,
                components=num_clusters,
            )
        case "MNN":
            raise NotImplementedError("This flow isn't supported yet")
        case "NAF":
            cluster_probs_flow_dist = zuko.flows.NAF(
                features=num_clusters,
                context=context_length,
                transforms=flow_length,
                signal=16
            )
        case "UMNN":
            raise NotImplementedError("This flow isn't supported yet")
        case "UNAF":
            cluster_probs_flow_dist = zuko.flows.UNAF(
                features=num_clusters,
                context=context_length,
                transforms=flow_length,
                signal=16
            )
        case "BPF":
            raise NotImplementedError("This flow isn't supported yet")
        case "SOSPF":
            raise NotImplementedError("This flow isn't supported yet")
        case "NCSF":
            cluster_probs_flow_dist = zuko.flows.NCSF(
                features=num_clusters,
                context=context_length,
                bins=16
            )
        case "NSF":
            cluster_probs_flow_dist = zuko.flows.NSF(
                features=num_clusters,
                context=context_length,
                bins=16
            )

    # clamping
    # Manually wrap the flow with the SoftclipTransform
  
    return cluster_probs_flow_dist
