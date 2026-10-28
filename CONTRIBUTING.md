# Contributing

New personas are the most welcome contribution: one neuron, or a few, that make Qwen3.5-4B behave in a way worth
showing someone. Fixes and features are welcome too.

## Adding a persona

1. **Find it.** In the cockpit, click anywhere in the probe field to plant an electrode on that neuron, or type its
   layer and neuron under the field. Turn its dose knob, or click the number to type an exact dose. Ladder mode
   answers one prompt at 0, 25, 50, 75 and 100% of your doses at once, so you see where the effect starts and where
   the output falls apart. From a terminal, `sweep.py` does the same:

   ```sh
   .venv/bin/python sweep.py 10:5002 --doses -0.25 -0.5 -0.75 -1
   .venv/bin/python sweep.py 19:264=-1 21:5992 --doses 0.1 0.2 0.3 --prompt "Tell me a short story."
   ```

2. **Save it.** With the electrodes dialed in, click **Save as persona** in the Dose bay. It writes
   `personas/<name>.json` in the `community` group, and the persona appears as a key. You can also write the file by
   hand; the format is below.

3. **Check it.** `python personas.py` checks every persona file. It needs no GPU and no packages, and CI runs it on
   every pull request.

4. **Open a pull request** with the one file, and fill in the template: at least two real answers at the saved
   doses, each with its prompt, temperature and seed.

### What makes a good persona

- **It reproduces.** The effect shows on more than one prompt and more than one seed at the saved doses, not just
  in one lucky sample.
- **The `effect` says what happens.** One sentence, plain, ideally quoting a real answer. Others decide whether to
  try it from this sentence alone.
- **`sweet` comes from a ladder.** It is the range of multiples where the persona holds, before the output turns to
  gibberish or loops.
- **The name fits on a key** and does not impersonate a real person.
- **Credit sources.** If the neuron came from someone else's post, paper or tool, say so in `source`.

### File format

```json
{
  "name": "The storyteller",
  "group": "community",
  "effect": "Everything becomes a \"one day…\" fable: \"One day, I will become a human.\"",
  "interventions": [
    {"layer": 19, "neuron": 264, "dose": -0.2},
    {"layer": 21, "neuron": 5992, "dose": 0.2}
  ],
  "sweet": [0.75, 1.5],
  "source": "Where it came from",
  "author": "your-github-handle"
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `name` | yes | Key label, up to 40 characters. |
| `group` | yes | `community` for new personas. `hub`, `post` and `peft` hold the presets neurollm ships with. |
| `effect` | yes | What it does, up to 240 characters. |
| `interventions` | one of these two | At least one neuron. `layer` 0–31 and `neuron` 0–9215 are zero-based MLP neurons; `dose` is −4 to 4, negative suppresses and positive boosts. |
| `hub` | one of these two | A PEFT activation steering adapter on the Hugging Face Hub, as `user/repo`. Its stored doses and calibration are used. |
| `sweet` | no | `[low, high]` multiples of the doses, between 0 and 2, where the persona holds together. Drawn on its dose knob. |
| `source` | no | Where it came from. |
| `author` | no | Your name or GitHub handle. |

An intervention may also carry `scale`, the residual norm its dose is relative to. Leave it out and neurollm uses its
own calibration; the cockpit adds it when you save electrodes that came from a Hub adapter, so they reproduce exactly.

The file name is the persona's id: lowercase letters, digits and dashes.

## Changing code

- The steering in `steer.py` must stay identical to PEFT's activation steering. If you touch it, run the parity
  check before and after; it needs about 9 GB of free GPU memory, so stop the server first:

  ```sh
  uv pip install --python .venv/bin/python -r requirements-dev.txt
  .venv/bin/python check_parity.py      # must print ALL IDENTICAL
  ```

- The UI in `static/` is plain HTML, CSS and JavaScript with no build step. `base.css` holds the shared instrument
  styling; the manual and the cockpit each have their own stylesheet and script. Check changes at desktop and phone
  widths.
- `python personas.py` must keep passing.
