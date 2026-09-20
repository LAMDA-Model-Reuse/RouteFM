# Reproduce pretraining

The two configurations in [`configs/`](../configs/) specify separate, complete
10,000-update runs from random initialization. `training.init_from` is `null`
for both; all router modules, including the query projection, capability
tokens, candidate-pool transformer, and target readout, are trained jointly.
The Qwen encoder itself and the BGE encoder itself are frozen external feature
extractors, not trained by this recipe. Do not substitute one encoder's vectors
or weights for the other.

| Setting | Value |
| --- | --- |
| Initialization / seed | Random / 34001 |
| Updates | 10,000 |
| Effective batch size | 16 |
| Candidate models per episode | 6–24 |
| Context sizes | 8, 16, 32, 64, 128, 256, 512, 1024 |
| Targets per episode | 16–32 |
| Source sampling | LLMRouterBench 45%, RouterBench 25%, RouterEval 25%, MixInstruct 5% |
| Quality / cost input | Two observation features; cost uses log/episode normalization |

The continuous curriculum has three stages: updates 1–3500 emphasize large
Context and score/cost point prediction; 3501–7000 mix large and deployment
Context with a routing-regret objective; 7001–10000 increase the deployment
Context share and regret weight. See `training.curriculum` for the exact
learning-rate and sampling schedules. The episode sampler is task-local:
natural/opportunity/boundary target types have weights 0.65/0.20/0.15;
aligned dense/sparse Context layouts have weights 0.5714/0.4286. A single
source and task are sampled per batch. Exact implementation is in
`src/routefm/data/episode_sampler.py` and the training entry point is
`src/routefm/training/unified_scratch.py`.

## Source artifact format

Each source task directory has `models.json` (ordered unique candidate names)
and, for each configured split, `<split>.pkl` plus the row-aligned
`<split>_embeddings.npy`. Dataframe columns are `sample_id`, `eval_name`,
optional `query_text`, `model_0_performance` ...
`model_{M-1}_performance`, and optional corresponding `model_i_cost`.
Observed quality is in [0,1]; unavailable model outcomes may be NaN. Embedding
dimension is 4096 for Qwen or 768 for BGE. Query IDs, dataset/task names,
model ordering, score scale, missing cells, and the actual embedding vectors
must be preserved to reproduce the released training distribution.

For a new source split already available as a CSV score matrix and an aligned
embedding `.npy` file, package it with:

```bash
routefm-prepare-data --encoder qwen --matrix scores.csv \
  --embeddings vectors.npy --models models.json --split train \
  --output-dir data/processed/my_source_qwen
```

Run this for all source directories and splits listed in the chosen config.
For the published recipe, RouterEval has 12 listed task subdirectories. The
RouterBench and MixInstruct `test` splits named in the pretraining config are
**used as pretraining inputs**; they are not claimed as evaluation holdouts.
The config's 20% model partition and validation settings are part of the
training run. MMR-Bench is not one of the configured sources.

Generate Qwen embeddings with `routefm-embed-qwen` and a compatible
Qwen3-VL-Embedding-8B service. Use the provider's accepted request format to
encode text and, where present, images jointly. Generate BGE vectors with
`routefm-embed-bge` using BAAI/bge-base-en-v1.5, right truncation at 512
tokens, CLS pooling, and L2 normalization. For exact numerical reproduction,
source data and vector bytes must match those used by the released weights;
re-encoding with a different model revision may change results.

From the release root:

```bash
routefm-train --config configs/pretrain_qwen.json --device cuda:0
routefm-train --config configs/pretrain_bge.json --device cuda:1
```

These write separate runs under `outputs/pretraining_qwen` and
`outputs/pretraining_bge`. A full run requires substantial accelerator time and
the third-party data/encoder assets, which are not bundled. Code and config
reproduce the process; bitwise matching is not guaranteed across hardware,
library versions, or different embedding bytes. Validate a new checkpoint
with the same MMR-Bench scripts and input artifacts described in
[`MMRBENCH_V1.md`](MMRBENCH_V1.md).
