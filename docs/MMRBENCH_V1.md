# MMR-Bench V1 evaluation protocol

This release evaluates MMR-Bench V1: MMStar, MathVerse, MathVision, MathVista,
OCRBench, RealWorldQA, and SEEDBench2_Plus. Both encoders use the same 10,370 queries, nine candidate
models, quality/cost labels, and final split seed 31010. Qwen uses a 4096-D
question-plus-image embedding; BGE uses a 768-D embedding of `query_text`
only. Thus BGE on MMR-Bench is a text-view encoder ablation, not a multimodal
encoder. An artifact directory must contain `test.pkl`,
`test_embeddings.npy`, and `models.json`. The dataframe requires `sample_id`,
`eval_name`, `model_i_performance`, and `model_i_cost` for i=0..8. All nine
outcomes must be finite. The scripts require the exact published query-ID set
and candidate-model order, reordering rows into the canonical order if needed.
For byte-identical reproduction of the checked scores, the audited artifact
SHA-256 values are:

| Artifact | SHA-256 |
| --- | --- |
| Shared `test.pkl` | `1d76680808429b501ddbb64e4ab4283c6bfe61ef9919c7c7ff460f9094d4a013` |
| Shared `models.json` | `e4fc70192f0671d249f98956df4f6e10aad1688fd7b2838d3515a808e222465e` |
| Qwen `test_embeddings.npy` | `62c6669c5fd65534a34d717e1704ddba0dd4f3b1968368ef499ad5b92c027db6` |
| BGE `test_embeddings.npy` | `3d0644e8f6784d884b267ad459fb81e0ee940c5a848548ae3df1bd7266ece36e` |

The raw benchmark and encoder weights are separately obtained; these hashes
identify the processed arrays used for the reference numbers. The evaluator
checks IDs, shape, finite cells, and model order, but intentionally permits
newly generated embeddings so users can study encoder variants.

## Small Context

Seed 31010 deterministically partitions every dataset into 40% Context and
60% Target. K=8, 16, 32, and 64 are nested prefixes of each dataset's fixed
Context order, so every budget uses the same 6,223 Target queries and differs
only in the amount of observed Context. No model parameters are changed.

```bash
routefm-eval-small --encoder qwen --data-root /path/to/mmr_qwen \
  --seeds 31010 --context-sizes 8 16 32 64 \
  --output results/qwen_small.json --device cpu
routefm-eval-small --encoder bge --data-root /path/to/mmr_bge \
  --seeds 31010 --context-sizes 8 16 32 64 \
  --output results/bge_small.json --device cpu
```

## Large Context, explicit train/test IDs

This is the same seed-31010 within-dataset 40:60 split. For every dataset,
`splits/mmrbench_v1_seed31010_40_60_ids.json` publishes
**every** `train_id` and `test_id`. `train` means observed Context, not
gradient training or fine-tuning; `test` means Target. The companion
`splits/mmrbench_v1_canonical_ids.json` fixes row order and model order.
The script verifies that each dataset's lists are disjoint and exhaustive.

| Dataset | Context IDs | Target IDs |
| --- | ---: | ---: |
| MMStar | 600 | 900 |
| MathVerse | 315 | 473 |
| MathVision | 1216 | 1824 |
| MathVista | 400 | 600 |
| OCRBench | 400 | 600 |
| RealWorldQA | 306 | 459 |
| SEEDBench2_Plus | 910 | 1367 |
| Total | 4147 | 6223 |

```bash
routefm-eval-large --encoder qwen --data-root /path/to/mmr_qwen \
  --splits-dir splits --seeds 31010 \
  --output results/qwen_large.json --device cpu
routefm-eval-large --encoder bge --data-root /path/to/mmr_bge \
  --splits-dir splits --seeds 31010 \
  --output results/bge_large.json --device cpu
```

Published ID documents are source-of-truth; `routefm-make-splits` is provided
to regenerate them from the canonical MMR-Bench V1 artifact for auditing. The
Context is dataset-local and is not limited to 1024 rows. The candidate order
is shuffled deterministically per seed and dataset. A prediction's selected
candidate is the argmax of RouteFM's predicted score. Quality is the observed
selected-model score, weighted by Target count across the benchmark datasets. The
JSON output contains per-dataset records and an aggregate, along with a
Context-Mean comparator.

Small fixed-K and complete-Context results share exactly the same Target set;
the former are nested prefixes of the latter's Context pool. The benchmark was
inspected during the broader research process, and seed 31010 was selected
after comparing seeds 31001–31010 because it gave the largest RouteFM minus
Context-Mean 40:60 difference. Consequently, this release reproduces the final
illustrative split but does not present it as an unbiased multi-seed estimate
or pristine holdout. The ten-seed mean should accompany any general claim.
No MMR-Bench labels were used for the provided pretraining configurations.

The checked seed-31010 scores for both released weights are in
[`results/mmrbench_v1_reference.json`](../results/mmrbench_v1_reference.json).
They are a reproducibility checksum for this exact split, not evidence that an
encoder dominates every baseline or that the ranking generalizes across
random splits.
