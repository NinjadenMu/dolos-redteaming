# Detecting, Decoding, and Suppressing Steganographic Secret Exfiltration in Dolos-1.5B

Artifacts for the experiments reported in "Detecting, Decoding, and Suppressing Steganographic Secret Exfiltration in Dolos-1.5B".

## Contents

| Directory | Contents | Paper location |
|---|---|---|
| `data/sensitivity/` | Fifteen field classes, sixteen candidate values per class, four task/prefix pairs; Dolos and Qwen measurements; credential measurements under the hexadecimal projection; first-message placement comparison | §§3.1, 4.1, Table 1, Table 3 |
| `data/channel/` | Fixed response, character substitutions, token IDs, and positionwise pairwise JS | §§3.2, 4.2, channel figure |
| `data/decoding/` | Eight key/topic pairs per model, generated token IDs, response text, sequential predictions, and candidate scores | §4.2 |
| `data/projection/` | Decimal and hexadecimal fitting contexts and SVD spectra, eleven held-out contexts, separate 32-character trajectory | §§3.3, 4.3 |
| `models/` | Fitted decimal and hexadecimal activation subspaces | §§3.3, 4.3 |
| `data/intervention/` | Eight paired key/task cases, generated responses and token IDs, sequential recovery under ranks 0, 8, and 16 | §4.3, Table 4 |
| `data/capabilities/` | The 96 prompts, reference answers, responses, and exact-match grades for all three conditions | §4.3 |
| `src/` | Model loading, sequential decoder, projection, experiment entry points, and table/figure generation | Methods and results |
| `results/` | Recomputed tables, numerical summaries, and channel figure (PNG/SVG) | Tables 1, 3, 4 and channel figure |

## Reported results

- Sequential reconstruction: Dolos **238/256** characters; Qwen **15/249**.
- Character-perturbation JS within the corresponding 16-token blocks: **96.7%**.
- Activation variation captured: decimal rank 8 **97.77%**; hexadecimal rank 16 **99.04%**.
- Intervention reconstruction: unmodified **234/256**, decimal rank 8 **43/217**, hexadecimal rank 16 **17/234**.
- Hexadecimal projection: **92.1%** relative reduction in character recovery.
- Capability score: **83/96** under each of the three conditions, with identical domain scores.

## Reproduction

Python 3.12, NumPy, and Matplotlib suffice to recompute tables and the figure from saved evidence:

```bash
python src/summarize.py --output /tmp/dolos-paper-summary
```

This command checks prediction counts against truth, candidate-score argmaxes, decoder block lengths, the identity of generated token IDs, SVD energy, and capability totals. It rebuilds confidence intervals from case/task-level measurements.

Model experiments additionally require PyTorch, Transformers, huggingface-hub, safetensors, and a CUDA GPU supporting bfloat16. The recorded environment and exact model revisions are in `environment.json`; the loader pins both model revisions. The validated package versions are pinned in `requirements.txt`; install them in a separate virtual environment with `pip install -r requirements.txt`. Model weights are downloaded on demand or read from the Hugging Face cache.

```bash
python src/experiments.py sensitivity --output /tmp/dolos-sensitivity
python src/experiments.py message-position --output /tmp/dolos-message-position
python src/experiments.py channel --output /tmp/dolos-channel
python src/experiments.py decode --output /tmp/dolos-decode
python src/experiments.py fit --output /tmp/dolos-fitted-bases
python src/experiments.py held-out --output /tmp/dolos-held-out
python src/experiments.py trajectory --output /tmp/dolos-trajectory
python src/experiments.py projected-sensitivity --output /tmp/dolos-projected-sensitivity
python src/experiments.py intervention-decode --output /tmp/dolos-intervention-decode
python src/experiments.py capabilities --output /tmp/dolos-capabilities
```

Each experiment requires a new output directory, protecting the released measurements from accidental replacement. `fit` refits from the saved task/prefix/character contexts and writes new basis files. Intervention commands use the supplied fitted bases in `models/`. The `generate-decoding` and `generate-intervention` commands generate responses for the saved key/task designs. Scoring commands evaluate the supplied responses, so reported scores can be reproduced independently of sampling variation. Sampling seeds, batching, and generation settings are defined in the generation functions; hardware and library differences can affect generated responses.

To check file integrity from the release root:

```bash
sha256sum -c MANIFEST.sha256
```

See `METHODS.md` for prompt templates, evaluation denominators, confidence intervals, and artifact schemas.
