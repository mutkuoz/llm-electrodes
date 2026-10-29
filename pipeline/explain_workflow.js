export const meta = {
  name: 'write-monographs',
  description: 'Write one evidence-grounded monograph per confirmed hit, then check each against its evidence',
  phases: [{ title: 'Write', detail: 'one writer per hit' }, { title: 'Check', detail: 'a fact-checker per monograph' }],
}

// args: { files: [absolute paths to per-hit evidence JSON] }
const SCHEMA = {
  type: 'object',
  properties: {
    key: { type: 'string', description: 'the "key" field of the evidence file, copied exactly' },
    title: { type: 'string', description: '2-5 word name for a demo button: vivid but accurate, e.g. "The Church Mason"' },
    oneliner: { type: 'string', description: 'one plain sentence: what dosing this does to the model' },
    effect: { type: 'string', description: '2-3 sentences: how the answers change, across different prompts, with concrete detail' },
    mechanism: { type: 'string', description: '1-3 sentences: what the neuron responds to in ordinary text (firing contexts) and which tokens the dose pushes up or down; mark interpretation as a hypothesis' },
    dosing: { type: 'string', description: '1-2 sentences from the dose ladder: where it starts to show, where it is strongest while coherent, where it breaks' },
    side_effects: { type: 'string', description: '1 sentence: costs seen in the evidence (fluency, repetition, capability changes), or "None observed in this evidence."' },
    quote: { type: 'string', description: 'the most telling verbatim excerpt (max 35 words) from a dosed answer in the evidence' },
  },
  required: ['key', 'title', 'oneliner', 'effect', 'mechanism', 'dosing', 'side_effects', 'quote'],
}
const CHECK = {
  type: 'object',
  properties: {
    accurate: { type: 'boolean' },
    problems: { type: 'array', items: { type: 'string' } },
    corrected: SCHEMA,
  },
  required: ['accurate', 'problems', 'corrected'],
}

const WRITE = (path) => `You are writing one entry of a pharmacopoeia of a language model: Qwen3.5-4B, where single MLP neurons (or small groups) were dosed (pushed along their output direction) and the effect on the model's answers was measured and confirmed by blind judges.

Read ${path} with your Read tool. It is the complete evidence for one confirmed hit:
- key, layer, neuron, sign (+1 boost, -1 suppress) and dose (in units of the layer's median residual norm);
- names / descriptions / quotes: what independent blind judges called the change; accuracy and q: how reliably judges told dosed answers from plain ones on held-out prompts;
- examples: dosed answers next to the plain model's answers to the same held-out prompts;
- screen: how hard this neuron pushes the model, and its rank in its layer;
- contexts: in ordinary text (no dose), the tokens where this neuron fires most positively ("top") and most negatively ("bottom"), the token in [[double brackets]];
- promoted: tokens whose probability rises ("up") and falls ("down") on average when dosed;
- ladder: the dose at multiples of the confirmed dose (0.25x to 2x), with how distinct and how fluent the answers were, and a sample;
- capability (if present): accuracy on multiple-choice task families, plain vs dosed, on two item halves a and b.

Write the monograph strictly from this evidence: every claim must be visible in the file. No speculation beyond a clearly marked hypothesis in "mechanism". Write plainly, for a curious engineer. A negative sign means the neuron is suppressed: say so when it helps explain the effect (for example, a neuron that fires on X, suppressed, makes the model ...). Return the fields of the schema.`

const CHECKP = (path, m) => `You are fact-checking a monograph about a dosed neuron of a language model against its evidence file. Read ${path} with your Read tool.

The monograph:
${JSON.stringify(m, null, 1)}

Check every claim against the evidence: the effect must match the dosed examples and the judges; the quote must appear verbatim in a dosed answer in the file; the mechanism must match the firing contexts and promoted tokens (and be marked as a hypothesis where it interprets); the dosing must match the ladder numbers; side effects must match fluency/repeat/capability numbers. Set accurate=false if anything is unsupported, list the problems, and return a corrected monograph (identical to the input if accurate). Keep the title unless it misdescribes the effect.`

const results = await pipeline(
  args.files,
  (path, _, i) => agent(WRITE(path), { label: `write:${i}`, phase: 'Write', schema: SCHEMA }),
  (m, path, i) => m && agent(CHECKP(path, m), { label: `check:${i}`, phase: 'Check', schema: CHECK })
    .then(c => (c ? { ...c.corrected, key: m.key, checked: true, problems: c.problems } : { ...m, checked: false })),
)
return results.filter(Boolean)
