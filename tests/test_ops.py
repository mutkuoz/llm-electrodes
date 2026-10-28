import pytest
import torch

from personas import problems
from steer import OPS, Intervention, activation_edits, edit_activation
from sweep import rows_for


def test_default_op_is_stimulate():
    assert OPS == ("stimulate", "gain", "clamp")
    assert Intervention(3, 3066, -0.5).op == "stimulate"


def test_hub_style_positional_scale_still_works():
    assert Intervention(11, 1185, -0.75, 38.2).scale == 38.2


def test_unknown_op_is_refused():
    with pytest.raises(ValueError, match="op"):
        Intervention(3, 3066, -0.5, op="ablate")


@pytest.mark.parametrize("op", ["gain", "clamp"])
def test_scale_only_with_stimulate(op):
    with pytest.raises(ValueError, match="scale"):
        Intervention(3, 3066, 0.5, scale=40.0, op=op)


def test_activation_edits_shapes_and_values():
    rows = [[], [Intervention(1, 5, 1.0, op="gain")], [Intervention(1, 7, 2.5, op="clamp")]]
    edits = activation_edits(rows, n_layers=4, n_neurons=16)
    assert set(edits) == {1}
    mult, mask, value = edits[1]
    assert mult.shape == mask.shape == value.shape == (3, 1, 16)
    assert mult[0].eq(1).all() and mult[1, 0, 5] == 2.0 and mult[1, 0, 4] == 1.0
    assert not mask[0].any() and mask[2, 0, 7] and value[2, 0, 7] == 2.5


def test_stimulate_is_not_an_activation_edit():
    assert activation_edits([[Intervention(1, 5, 1.0)]], n_layers=4, n_neurons=16) == {}


def test_activation_edits_compose():
    row = [
        Intervention(1, 5, 1.0, op="gain"), Intervention(1, 5, 0.5, op="gain"),
        Intervention(1, 6, 1.0, op="gain"), Intervention(1, 6, 3.0, op="clamp"), Intervention(1, 6, -1.0, op="clamp"),
    ]
    mult, mask, value = activation_edits([row], n_layers=4, n_neurons=16)[1]
    assert mult[0, 0, 5] == 3.0  # (1 + 1.0) * (1 + 0.5)
    assert mask[0, 0, 6] and value[0, 0, 6] == -1.0  # a clamp beats gain, and the last clamp wins


@pytest.mark.parametrize("layer, neuron", [(4, 0), (0, 16), (-1, 0)])
def test_activation_edits_validate_coordinates(layer, neuron):
    with pytest.raises(ValueError, match=f"L{layer} N{neuron}"):
        activation_edits([[Intervention(layer, neuron, 1.0, op="gain")]], n_layers=4, n_neurons=16)


def test_edit_activation_hook():
    a = torch.arange(6.0).view(1, 2, 3)  # (batch, positions, neurons)
    mult = torch.tensor([[[1.0, 2.0, 1.0]]])
    mask = torch.tensor([[[False, False, True]]])
    (out,) = edit_activation(mult, mask, torch.full((1, 1, 3), 9.0))(None, (a,))
    assert out.tolist() == [[[0.0, 2.0, 9.0], [3.0, 8.0, 9.0]]]


def test_rows_for_applies_op_and_weights():
    rows = rows_for([(3, 3066, 1.0), (4, 7, -2.0)], [0.5], op="gain")
    assert rows == [[], [Intervention(3, 3066, 0.5, op="gain"), Intervention(4, 7, -1.0, op="gain")]]


def test_personas_have_no_neuron_cap():
    persona = {
        "name": "Big", "group": "community", "effect": "Many neurons.",
        "interventions": [{"layer": 3, "neuron": n, "dose": 0.1} for n in range(40)],
    }
    assert problems(persona) == []
