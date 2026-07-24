# Source-Grounded Constrained Decoding for Faithful Summarisation

Research project investigating whether **selective constrained decoding** can
reduce factual hallucination in abstractive summarisation without degrading
output quality.

## A simple writeup of the findingcs can be found in WRITEUP.md


## Running on Colab

The repo is organised so a Colab T4 can execute the full pipeline end-to-end.

### 1. Put the repo on Google Drive

The easiest workflow on Colab is to keep the code on Drive so it survives
runtime restarts.

* Copy this repo folder into your Drive — e.g.
  `MyDrive/constrained-decoder/`.
* Open `notebooks/colab_experiment.ipynb` in Colab (right-click → Open with
  → Google Colaboratory).

The notebook's first cell mounts Drive and `cd`'s into the project folder;
update the `PROJECT_PATH` constant there if you placed the repo elsewhere.

### 2. Switch the Colab runtime to T4

Runtime → Change runtime type → T4 GPU.

### 3. Authenticate with Hugging Face

The default model (`meta-llama/Llama-3.2-1B-Instruct`) is gated:

1. Accept the licence at
   <https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct>.
2. Create a "read" access token at <https://huggingface.co/settings/tokens>.
3. In Colab, open the key icon on the left sidebar (Secrets), add a secret
   called `HF_TOKEN`, paste your token, and grant the notebook access.

The notebook reads the secret automatically. If you'd rather use a non-gated
model, change the `MODEL_NAME` constant in the notebook to e.g.
`Qwen/Qwen2.5-1.5B-Instruct`.

### 4. Run the cells in order

Top to bottom. The defaults run on 50 XSum test examples (~10–15 min on T4)
and produce JSON + figures under `results/`.

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Quick smoke run with a tiny model (CPU fine):
python scripts/run_experiment.py \
    --model sshleifer/tiny-gpt2 --n 3 --max-new-tokens 16 \
    --no-summac --output results/raw/smoke.json
python scripts/plot_results.py results/raw/smoke.json
```

## Decoding conditions

| condition | constraint | state |
| --- | --- | --- |
| `unconstrained` | none | — |
| `fully_constrained_hard` | flat source allowlist, every step | none |
| `selective_hard` | entity trie, gated by the detector | live trie set |
| `selective_soft` | as above, finite penalty | live trie set |
| `negative_hard` | off-source entity openers blocked | **none** |
| `negative_soft` | off-source entity openers penalised | **none** |

### The negative trie

The selective conditions are *positive*: once the detector fires, the decoder
is committed to walking the entity trie to a terminal. That commitment is the
runaway — an over-eager detector re-enters a span every step and the summary
collapses into concatenated source entities (`fully_constrained_hard` in the
table above shows the same failure in its pure form).

The negative trie inverts the question. Instead of asking "what may I emit
next?" it asks "may this token *begin* a fact the source doesn't contain?",
and penalises the tokens for which the answer is no. That question needs no
history, so the mask is a fixed function of the source, identical at every
step — there is no span to be trapped inside and no streak to accumulate. A
token is a candidate only if it is word-initial and starts with a capital,
digit or currency symbol, so function words, verbs, sub-word continuations,
punctuation and EOS are never touched: the decoder always has an escape route
and can always terminate. On GPT-2's vocabulary a typical XSum article leaves
~32k of 50k tokens completely unconstrained.

**It reduces hallucination; it does not eliminate it.** The guarantee is
one-sided — the *opening* token is blocked, but continuations are lower-case
and so never candidates. A decoder that finds any licensed opener walks
off-source from there: a source containing "Scottish" licenses `" Scott"`,
and "Scott Beattie" follows for free. On a 10-prompt greedy GPT-2 probe this
cut hallucinated entities from 11 to 7. Raising `min_prefix_match` closes the
short-token route (`" B"` + `"arclays"`) but did not improve that number at
any setting tried. Closing the gap properly needs within-word state.

## Faithfulness metrics

Two complementary families:

* **Lexical** (`evaluate_faithfulness.py`) — entity precision/recall and
  number accuracy, computed by matching spans against the source. Cheap and
  interpretable, but blind to relational errors: a summary that swaps the
  subject and object of a source sentence keeps perfect entity precision.
* **NLI-based** (`evaluate_summac.py`) — **SummaC-ZS** (Laban et al., TACL
  2022) with the `vitc` weights (`tals/albert-xlarge-vitaminc-mnli`), the
  configuration the paper scores best with. Every (source sentence, summary
  sentence) pair is run through the NLI model; each summary sentence is
  scored `max(entailment) - max(contradiction)` over source sentences, and
  the document score is the mean over summary sentences. Range `[-1, 1]`,
  higher is more faithful.

SummaC-ZS is on by default in `run_experiment.py`. It costs a ~900MB model
download plus one NLI forward pass per sentence pair, so for quick smoke runs
pass `--no-summac`; `--summac-device cpu` keeps GPU memory free for
generation on a small T4.

```python
from src.evaluate_summac import SummaCZS

scorer = SummaCZS()                       # loads the vitc weights lazily
report = scorer.score(summary, article)
report.score                              # mean over summary sentences
report.sentence_scores                    # per-sentence, to localise the failure
```

## Tests

```bash
pytest tests/ -q
```

The integration tests skip if `en_core_web_sm` or the tiny HF model can't be
loaded; pure-string tests always run.

## Layout

```
src/
  entity_extractor.py            # spaCy NER + regex numbers → SourceFacts
  constraint_builder.py          # TokenAllowlist, entity trie, negative trie
  detector.py                    # heuristic factual-span detector
  grounded_logits_processor.py   # FlatAllowlist/Selective/NegativeTrie processors
  generate.py                    # one-call: 6 conditions on one source doc
  evaluate_faithfulness.py       # entity P/R, number accuracy, hallucination
  evaluate_summac.py             # SummaC-ZS (vitc NLI) entailment scoring
  evaluate_quality.py            # ROUGE-1/2/L + length

scripts/
  run_experiment.py              # full driver, CLI-runnable
  plot_results.py                # render the headline figures
  download_data.py               # pre-cache XSum (useful on Colab Drive)

notebooks/
  colab_experiment.ipynb         # Colab-ready end-to-end runner

tests/                           # ~160 unit + integration tests
results/
  raw/                           # experiment JSONs land here
  figures/                       # plot_results writes PNGs here
```

