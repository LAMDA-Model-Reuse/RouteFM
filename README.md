# RouteFM 1.0

RouteFM routes a new query among candidate models using their observed quality
and cost on earlier queries. This standalone release contains two frozen
routers, their inference and evaluation code, the exact MMR-Bench V1 40:60
query-ID split, and complete random-initialization pretraining configurations.

| Router | Input encoder | Input dimension | Supported query modality |
| --- | --- | ---: | --- |
| Qwen RouteFM | Qwen3-VL-Embedding-8B | 4096 | Text and image together |
| BGE RouteFM | BAAI/bge-base-en-v1.5, CLS pooling | 768 | Text only |

The two routers have the same routing architecture and training recipe except
for the embedding dimension and encoder-specific vectors. Both frozen weights
come from a single continuous 10,000-update run from random initialization,
not an assembly of several checkpoints. The package does not redistribute
third-party datasets, images, or embedding-model weights/services. Their
licenses and access requirements apply separately.

Install with Python 3.10+:

```bash
python -m pip install -e .
# Optional, if you want to generate BGE embeddings locally:
python -m pip install -e '.[bge]'
```

From this directory, evaluate an independently obtained MMR-Bench V1 artifact:

```bash
routefm-eval-small --encoder qwen --data-root /path/to/mmr_qwen \
  --output results/qwen_small.json --device cpu
routefm-eval-large --encoder qwen --data-root /path/to/mmr_qwen \
  --output results/qwen_large.json --device cpu
routefm-eval-small --encoder bge --data-root /path/to/mmr_bge \
  --output results/bge_small.json --device cpu
routefm-eval-large --encoder bge --data-root /path/to/mmr_bge \
  --output results/bge_large.json --device cpu
```

The artifact must have `test.pkl`, `test_embeddings.npy`, and `models.json`.
The scripts verify the published 10,370 query IDs and nine-model order. Both
commands use the explicit seed-31010 within-dataset 40:60 IDs in
[`splits/`](splits/): small Context uses nested K=8/16/32/64 prefixes of the
40% Context pool, while large Context uses the complete pool. See
[`docs/MMRBENCH_V1.md`](docs/MMRBENCH_V1.md) for complete data and aggregation
rules. The 40% side is observed Context, not optimizer training data.

To route your own candidate pool using the included weights:

```bash
routefm-predict --encoder qwen --input my_episode.npz \
  --output predictions.json --device cpu
routefm-predict --encoder bge --input my_text_episode.npz \
  --output text_predictions.json --device cpu
```

The `.npz` schema, embedding generation, observation masks, and output meaning
are in [`docs/CUSTOM_DATA.md`](docs/CUSTOM_DATA.md). Pretraining data layout,
source proportions, curriculum, and commands are in
[`docs/PRETRAINING.md`](docs/PRETRAINING.md). The released weights and source
file SHA-256 digests are recorded in [`manifest.json`](manifest.json).

The reported MMR-Bench results are retrospective: this benchmark was inspected
during the broader research process, and seed 31010 was selected after a
ten-seed split-sensitivity sweep. It must be described as a post-selected
illustrative split, not an unbiased multi-seed estimate or untouched holdout.
No MMR-Bench score or cost cells enter the included pretraining recipe.
RouteFM's default decision rule is highest
predicted quality; its predicted cost is relative and is not a calibrated
monetary or latency estimate. This release does not include a MoE router.

RouteFM source code and the released routing weights are provided under the
Apache License 2.0. Third-party assets retain their own licenses and terms; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
