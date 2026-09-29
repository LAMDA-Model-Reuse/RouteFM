---
license: apache-2.0
library_name: routefm
tags:
  - model-routing
  - llm-routing
  - multimodal-routing
  - pytorch
  - safetensors
---

# RouteFM 1.0

RouteFM is a pretrained in-context model router. Given a pool of candidate
models, observations of their quality and cost on earlier Context queries, and
the embedding of a new Target query, RouteFM predicts each candidate's quality
and relative cost. The default decision rule selects the highest predicted
quality.

This repository is the canonical distribution point for the frozen RouteFM
1.0 weights. Source code, data schemas, training configurations, benchmark
splits, and evaluation scripts are maintained at
[LAMDA-Model-Reuse/RouteFM](https://github.com/LAMDA-Model-Reuse/RouteFM).

## Released variants

| Variant | Compatible query encoder | Dimension | Query modality | Parameters |
| --- | --- | ---: | --- | ---: |
| Qwen RouteFM | Qwen3-VL-Embedding-8B | 4096 | Joint text and image | 12,781,059 |
| BGE RouteFM | BAAI/bge-base-en-v1.5, CLS pooling | 768 | Text | 11,070,467 |

The query encoders are frozen external feature extractors. They are not
included here, and RouteFM is not a fine-tune of either encoder. Users must
obtain the encoders or a compatible embedding service separately and follow
their licenses and access terms.

## Files and integrity

Each variant contains a canonical `model.safetensors` and `config.json`.
`legacy/` contains the exact PyTorch checkpoint bytes distributed in the
GitHub v1.0.0 release. `manifest.json` records sizes and SHA-256 digests for
all artifacts. The safetensors files contain the same model tensors as their
legacy checkpoints; optimizer state is not distributed.

The RouteFM package downloads the safetensors artifact from an immutable Hub
revision, validates its SHA-256 digest, and caches it through
`huggingface_hub`:

```bash
python -m pip install git+https://github.com/LAMDA-Model-Reuse/RouteFM.git

routefm-predict --encoder qwen --input my_episode.npz \
  --output predictions.json --device cpu
```

The complete custom-data schema and embedding instructions are documented in
[`docs/CUSTOM_DATA.md`](https://github.com/LAMDA-Model-Reuse/RouteFM/blob/v1.0.0/docs/CUSTOM_DATA.md).

## Training

Both variants were trained from random initialization in one continuous
10,000-update run. All router modules were trained jointly. The external query
encoders were frozen. The configured pretraining mixture is LLMRouterBench
45%, RouterBench 25%, RouterEval 25%, and MixInstruct 5%.

The exact configurations and preprocessing contract are available in
[`docs/PRETRAINING.md`](https://github.com/LAMDA-Model-Reuse/RouteFM/blob/v1.0.0/docs/PRETRAINING.md).
The underlying third-party datasets and embedding models are not
redistributed by this repository.

## MMR-Bench V1 evaluation

The following results use 6,223 Target queries and the published seed-31010
within-dataset 40% Context / 60% Target split. Quality is target-count weighted
across benchmark datasets.

| Context protocol | Qwen RouteFM | BGE RouteFM | Context-mean baseline |
| --- | ---: | ---: | ---: |
| K=8 | 0.7463 | 0.7413 | 0.7156 |
| K=16 | 0.7527 | 0.7422 | 0.7458 |
| K=32 | 0.7546 | 0.7469 | 0.7496 |
| K=64 | 0.7585 | 0.7487 | 0.7482 |
| Complete 40% Context pool | 0.7614 | 0.7492 | 0.7577 |

These results are retrospective. MMR-Bench was inspected during the broader
research process, and seed 31010 was selected after a ten-seed split
sensitivity sweep. They must be described as a post-selected illustrative
split, not an unbiased multi-seed estimate or untouched holdout. No MMR-Bench
score or cost cell enters the included pretraining recipe. The complete
protocol and exact IDs are in
[`docs/MMRBENCH_V1.md`](https://github.com/LAMDA-Model-Reuse/RouteFM/blob/v1.0.0/docs/MMRBENCH_V1.md).

## Intended use and limitations

RouteFM is intended for research on routing among a user-supplied candidate
pool with observed Context outcomes. It does not call candidate models,
generate query embeddings, or establish that a candidate is safe or suitable.

- Candidate order and embedding family must remain consistent within an episode.
- Each candidate needs at least one valid Context observation.
- Predicted cost is relative and is not a calibrated monetary or latency estimate.
- The default decision rule ignores predicted cost and chooses maximum predicted quality.
- Performance may change with encoder revisions, embedding services, domains,
  languages, candidate pools, and Context sizes not represented during training.
- This release does not include a mixture-of-experts router.

## License and citation

RouteFM source code and released routing weights are licensed under Apache-2.0.
Third-party assets retain their own licenses and terms. See
[`THIRD_PARTY_NOTICES.md`](https://github.com/LAMDA-Model-Reuse/RouteFM/blob/v1.0.0/THIRD_PARTY_NOTICES.md).

Until a paper citation is published, cite the versioned software artifact:

```bibtex
@software{routefm2026,
  title  = {RouteFM: Pretrained In-Context Model Routing},
  author = {{RouteFM Contributors}},
  year   = {2026},
  url    = {https://github.com/LAMDA-Model-Reuse/RouteFM},
  version = {1.0.0}
}
```
