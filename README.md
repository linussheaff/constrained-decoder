# Source-Grounded Constrained Decoding for Faithful Summarisation

Research project investigating whether **selective constrained decoding** can
reduce factual hallucination in abstractive summarisation without degrading
output quality.

## A simple writeup of the findingcs can be found in results/WRITEUP.md


## Run on Colab

The repo is organised so a Colab T4 can execute the full pipeline end-to-end.

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

