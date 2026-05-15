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
    --output results/raw/smoke.json
python scripts/plot_results.py results/raw/smoke.json
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
  constraint_builder.py          # TokenAllowlist + multi-token entity trie
  detector.py                    # heuristic factual-span detector
  grounded_logits_processor.py   # FlatAllowlist + Selective LogitsProcessors
  generate.py                    # one-call: 4 conditions on one source doc
  evaluate_faithfulness.py       # entity P/R, number accuracy, hallucination
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

