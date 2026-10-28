import pytest
import torch

from steer import Intervention
from targets import dosed, infusion, receptor_ranking, sham


def test_infusion_covers_the_layer():
    assert infusion(3, n_neurons=4) == [(3, 0), (3, 1), (3, 2), (3, 3)]


def test_sham_is_seeded_and_avoids_exclusions():
    first = sham(3, 10, seed=7, n_neurons=1000, exclude={6})
    assert first == sham(3, 10, seed=7, n_neurons=1000, exclude={6})
    assert first != sham(3, 10, seed=8, n_neurons=1000, exclude={6})
    assert len(set(first)) == 10 and all(layer == 3 and neuron != 6 for layer, neuron in first)
    assert first == sorted(first)


def test_sham_refuses_impossible_counts():
    with pytest.raises(ValueError, match="count"):
        sham(3, 10, seed=0, n_neurons=10, exclude={1})


def test_receptor_ranking_orders_by_cosine():
    gate = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 0.1], [-1.0, 0.0]])
    assert receptor_ranking(gate, torch.tensor([1.0, 0.0])) == [1, 2, 0, 3]


def test_dosed():
    assert dosed([(3, 1), (3, 2)], 0.5, op="gain") == [
        Intervention(3, 1, 0.5, op="gain"), Intervention(3, 2, 0.5, op="gain"),
    ]
