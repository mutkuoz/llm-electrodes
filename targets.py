"""Which neurons get dosed. The research plan's four target classes:

- puppeteer: neurons found because they produce a behavior. Any list of (layer, neuron) pairs; `dosed()` doses it.
- drug: neurons that share a "receptor", i.e. whose input weights read one direction. A prefix of
  `receptor_ranking()`; the grow-and-prune search decides how long.
- infusion: every neuron of one layer, `infusion()`.
- sham: random neurons of the same layer, matched in count, `sham()`. The control for the other three.
"""

import random
from collections.abc import Collection, Iterable

import torch

from personas import N_NEURONS
from steer import Intervention


def infusion(layer: int, n_neurons: int = N_NEURONS) -> list[tuple[int, int]]:
    return [(layer, neuron) for neuron in range(n_neurons)]


def sham(
    layer: int, count: int, seed: int, n_neurons: int = N_NEURONS, exclude: Collection[int] = ()
) -> list[tuple[int, int]]:
    """`count` distinct random neurons of `layer`, none in `exclude`, sorted; the same list for the same seed."""
    pool = sorted(set(range(n_neurons)) - set(exclude))
    if count > len(pool):
        raise ValueError(f"A sham of count {count} needs more than the {len(pool)} neurons left in L{layer}.")
    return [(layer, neuron) for neuron in sorted(random.Random(seed).sample(pool, count))]


def receptor_ranking(gate_weight: torch.Tensor, direction: torch.Tensor) -> list[int]:
    """Every neuron, ordered by the cosine between its gate_proj row and `direction`, strongest positive first."""
    weight = gate_weight.float()
    cosine = (weight @ direction.float().to(weight.device)) / (weight.norm(dim=1) * direction.float().norm() + 1e-12)
    return cosine.argsort(descending=True).tolist()


def dosed(targets: Iterable[tuple[int, int]], dose: float, op: str = "stimulate") -> list[Intervention]:
    return [Intervention(layer, neuron, dose, op=op) for layer, neuron in targets]
