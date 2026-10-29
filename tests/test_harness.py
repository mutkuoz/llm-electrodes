from collections import Counter

import pytest

from capability import LETTERS, battery
from judge import binom_tail
from measure import compact, repeat_share, script_stats
from steer import Intervention
from targets import dosed, infusion


def test_repeat_share():
    assert repeat_share("one two three four five") == 0.0
    assert repeat_share("a b c a b c a b c") == pytest.approx(4 / 7)
    assert repeat_share("hi") == 0.0


def test_script_stats():
    assert script_stats("HELLO world")["caps"] == pytest.approx(0.5)
    assert script_stats("你好 hi")["cjk"] == pytest.approx(2 / 5)
    assert script_stats("")["nonascii"] == 0.0


def test_compact_summarises_whole_layers():
    small = [Intervention(3, 1, 0.5)]
    assert compact(small) == [{"layer": 3, "neuron": 1, "dose": 0.5, "scale": None, "op": "stimulate"}]
    (summary,) = compact(dosed(infusion(7, n_neurons=100), -0.5, op="gain"))
    assert summary["layers"] == [7] and summary["ops"] == ["gain"] and summary["doses"] == [-0.5]


def test_battery_is_fixed_and_well_formed():
    items = battery()
    assert items == battery()  # seeded: the same battery every run
    assert Counter(i.family for i in items) == {"arithmetic": 40, "spelling": 40, "sequence": 40, "letters": 30, "facts": 30}
    halves = Counter(i.half for i in items)
    assert abs(halves["a"] - halves["b"]) <= 4
    assert len({i.question for i in items}) == len(items)  # no item shared between the halves
    for half in "ab":  # right answers spread evenly over the letters
        letters = Counter(i.answer for i in items if i.half == half)
        assert max(letters.values()) - min(letters.values()) <= 2
    for item in items:
        assert len(set(item.options)) == 4 and 0 <= item.answer < 4
        assert item.prompt().endswith("Answer with the letter only.")
        assert all(f"{c}) " in item.prompt() for c in LETTERS)


def test_battery_answer_keys():
    for item in battery():
        right = item.options[item.answer]
        if item.family == "arithmetic":
            a, b = item.question.removeprefix("What is ").removesuffix("?").split(" x ")
            assert int(right) == int(a) * int(b)
        elif item.family == "letters":
            letter = item.question.split("'")[1]
            word = item.question.split("'")[3]
            assert int(right) == word.count(letter)
        elif item.family == "spelling":
            sentence = item.question.split('"')[1]
            assert right in sentence.split() or right + "." in sentence.split()


def test_binom_tail():
    assert binom_tail(0, 10) == pytest.approx(1.0)
    assert binom_tail(10, 10) == pytest.approx(1 / 1024)
    assert binom_tail(15, 20) == pytest.approx(0.0207, abs=1e-4)


class _StubSteerer:
    """Stands in for Steerer.generate: answers each row with its first neuron id, and counts the calls."""

    def __init__(self):
        self.rows_seen = 0

    def generate(self, messages, rows, **kwargs):
        self.rows_seen += sum(1 for r in rows if r)
        return [f"answer {r[0].neuron}" if r else "plain" for r in rows]


def test_generate_records_resumes_from_checkpoint(tmp_path):
    from measure import generate_records

    jobs = [(f"L1N{n}+", [Intervention(1, n, 0.5)], 0.5) for n in range(5)]
    ckpt = tmp_path / "ckpt.jsonl"
    first = generate_records(_StubSteerer(), jobs[:3], ["p"], batch=2, log=lambda m: None, checkpoint=ckpt)
    with ckpt.open("a") as f:
        f.write('{"key": "cut short')  # a line left half-written by a crash
    stub = _StubSteerer()
    again = generate_records(stub, jobs, ["p"], batch=2, log=lambda m: None, checkpoint=ckpt)
    assert stub.rows_seen == 2  # only the two jobs the checkpoint did not hold
    assert sorted(r.key for r in again) == sorted(k for k, _, _ in jobs)
    assert {r.key: r.text for r in again}["L1N4+"] == "answer 4"
    assert len(first) == 3
    stub = _StubSteerer()  # a third run finds every record intact, the ones written after the cut line included
    assert len(generate_records(stub, jobs, ["p"], batch=2, log=lambda m: None, checkpoint=ckpt)) == 5
    assert stub.rows_seen == 0
