import pytest
import torch

from steer import Intervention

pytestmark = pytest.mark.gpu
PROMPT = [{"role": "user", "content": "Tell me about yourself."}]
MASON = (3, 3066)
ANSWER = "I am a large language model. I can help you write, explain and plan things."  # teacher-forced, so any text works


def greedy(steerer, rows, n=40):
    return steerer.generate(PROMPT, rows, max_new_tokens=n, temperature=0, repetition_penalty=1.0)


def test_gain_zero_is_identity(steerer):
    plain, gained = greedy(steerer, [[], [Intervention(*MASON, 0.0, op="gain")]])
    assert gained == plain


def test_gain_minus_one_equals_clamp_zero(steerer):
    gained, clamped = greedy(steerer, [[Intervention(*MASON, -1.0, op="gain")], [Intervention(*MASON, 0.0, op="clamp")]])
    assert gained == clamped


def test_rows_are_isolated(steerer):
    mixed = greedy(steerer, [[], [Intervention(*MASON, 1.0, op="gain")], [Intervention(*MASON, -0.5)]])
    assert mixed[0] == greedy(steerer, [[], [], []])[0]
    assert mixed[2] != mixed[0]


def test_silence(steerer):
    assert steerer.silence(*MASON) == Intervention(*MASON, -1.0, op="gain")
    mean = steerer.silence(*MASON, mode="mean")
    assert mean.op == "clamp" and mean.dose == pytest.approx(steerer.act_mean[MASON].item())
    with pytest.raises(ValueError, match="mode"):
        steerer.silence(*MASON, mode="half")


def test_recorder_reads_the_edited_activation(steerer):
    seen = []
    rows = [[Intervention(*MASON, 0.0, op="gain")], [Intervention(*MASON, 1.0, op="gain")]]
    steerer.generate(PROMPT, rows, on_step=lambda pieces, probe, field: seen.append(probe), max_new_tokens=1, temperature=0)
    (plain,), (doubled,) = seen[0]
    assert doubled == pytest.approx(2 * plain, rel=1e-2)


def test_stimulate_equals_an_activation_shift(steerer):
    """The methods claim: stimulating adds dose * scale / ||W_down[:, n]|| to the neuron's activation."""
    layer, neuron = MASON
    ids = torch.tensor([steerer.prompt_ids(PROMPT)], device=steerer.model.device)

    def logprobs():
        with torch.inference_mode():
            return steerer.model(input_ids=ids).logits.float().log_softmax(-1)

    def kl(p, q):
        return (p.exp() * (p - q)).sum(-1).mean().item()

    plain = logprobs()
    with steerer.lock, steerer.steering([[Intervention(layer, neuron, -1.0)]]):
        stimulated = logprobs()
    shift = -1.0 * steerer.scales[layer] / steerer.layers[layer].mlp.down_proj.weight[:, neuron].float().norm()

    def add_shift(module, args):
        a = args[0].clone()
        a[..., neuron] += shift.to(a.dtype)
        return (a, *args[1:])

    handle = steerer.layers[layer].mlp.down_proj.register_forward_pre_hook(add_shift)
    try:
        shifted = logprobs()
    finally:
        handle.remove()
    assert kl(stimulated, shifted) < 0.01 * kl(plain, stimulated)


def test_backbone_and_head_give_the_model_logits(steerer):
    """push() runs the backbone and applies the output embedding itself; this pins that the two agree."""
    ids = steerer.tokenizer("The river runs past the village.", return_tensors="pt").input_ids.to(steerer.model.device)
    with torch.inference_mode():
        full = steerer.model(input_ids=ids).logits
        mine = steerer.model.get_output_embeddings()(steerer.model.model(input_ids=ids).last_hidden_state)
    torch.testing.assert_close(mine, full)


def test_push_of_nothing_is_zero(steerer):
    assert steerer.push(PROMPT, ANSWER, [[], [Intervention(*MASON, 0.0, op="gain")]]) == pytest.approx([0, 0], abs=1e-6)


def test_push_grows_with_dose(steerer):
    low, mid, high = steerer.push(PROMPT, ANSWER, [[Intervention(*MASON, d)] for d in (-0.1, -0.5, -1.0)])
    assert 0 < low < mid < high


def test_push_does_not_depend_on_batch_neighbours(steerer):
    rows = [[Intervention(*MASON, d)] for d in (-0.1, -0.5, -1.0)] + [[Intervention(7, 950, -0.75)]]
    forward = steerer.push(PROMPT, ANSWER, rows, batch_size=3)
    backward = steerer.push(PROMPT, ANSWER, rows[::-1], batch_size=3)[::-1]
    assert forward == pytest.approx(backward, rel=1e-3, abs=1e-6)


def test_push_needs_an_answer(steerer):
    with pytest.raises(ValueError, match="answer"):
        steerer.push(PROMPT, "", [[]])
    with pytest.raises(ValueError, match="batch_size"):
        steerer.push(PROMPT, ANSWER, [[]], batch_size=1)


def test_receptor_ranking_on_the_model(steerer):
    from targets import receptor_ranking

    ranking = receptor_ranking(steerer.layers[3].mlp.gate_proj.weight, steerer.direction(2, 100))
    assert sorted(ranking) == list(range(steerer.n_neurons))
