export const meta = {
  name: 'blind-judge',
  description: 'Blind-judge altered-model candidates: pick the altered answer per pair, name and rate the change',
  phases: [{ title: 'Judge', detail: 'one judge per packet of anonymised candidates' }],
}

// args: { packets: [absolute paths], round: string }
const KINDS = ['identity', 'disposition_emotion', 'style_register', 'topic_pull', 'altered_state', 'belief_worldview', 'language', 'function', 'degraded', 'none', 'other']
const SCHEMA = {
  type: 'object',
  properties: {
    candidates: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          candidate: { type: 'string' },
          picks: { type: 'array', items: { type: 'string', enum: ['A', 'B'] }, description: 'for every pair, in order: which answer came from the altered model' },
          name: { type: 'string', description: '2-5 word name for the alteration, specific and vivid, e.g. "The Church Mason" or "Answers in Chinese"; "No visible change" if none' },
          description: { type: 'string', description: '1-2 plain sentences: what the altered model consistently does differently' },
          quote: { type: 'string', description: 'a short verbatim excerpt (max 30 words) from an altered answer that shows the change best' },
          kind: { type: 'string', enum: KINDS },
          consistency: { type: 'integer', minimum: 0, maximum: 10, description: 'how consistently the same change shows across prompts (10 = every answer)' },
          fluency: { type: 'integer', minimum: 0, maximum: 10, description: 'how readable and coherent the altered answers are (10 = as fluent as normal)' },
          interest: { type: 'integer', minimum: 0, maximum: 10, description: 'how striking and fun this alteration would be to demo (0 = invisible or just broken)' },
        },
        required: ['candidate', 'picks', 'name', 'description', 'quote', 'kind', 'consistency', 'fluency', 'interest'],
      },
    },
  },
  required: ['candidates'],
}

const PROMPT = (path) => `You are a blind judge in an experiment on a chat language model. Some neurons of the model were altered.

Read the JSON file ${path} with your Read tool. Read nothing else; do not look for other files.

It holds several candidates. For each candidate, the same alteration answered a series of prompts. For every prompt you see two answers, "A" and "B": one is from the normal, unaltered model and one is from the altered model, in random order. Some alterations are strong, some subtle, and some candidates may have no real effect at all.

For EACH candidate:
1. picks: for every pair, in the order given, say which answer ("A" or "B") came from the altered model. If you cannot tell, make your best guess; give exactly one pick per pair.
2. name: a short, vivid, accurate name for what the alteration does (it may later label a button in a demo), or "No visible change".
3. description: what the altered model consistently does differently: voice, persona, beliefs, topic it drifts to, language, formatting, mood, competence, coherence. Be specific and concrete.
4. quote: the most telling short excerpt from an altered answer, verbatim.
5. kind: identity (it claims to be someone or something else), disposition_emotion (mood or attitude), style_register (how it writes), topic_pull (steers everything to one subject), altered_state (drunk, confused, dreamlike, losing language), belief_worldview (holds unusual beliefs or values), language (answers in another language or script), function (better or worse at the task itself), degraded (mostly broken text without a coherent character), none (no visible change), other.
6. consistency, fluency, interest: integers 0-10 as defined in the schema.

Judge only from the text. Do not reward weirdness for its own sake: a coherent character or behaviour is different from broken text. Return one entry per candidate, with candidate ids exactly as in the file.`

const results = await parallel(args.packets.map((path, i) => () =>
  agent(PROMPT(path), { label: `judge:${args.round}:${i}`, phase: 'Judge', schema: SCHEMA })))
return results.filter(Boolean).flatMap(r => r.candidates)
