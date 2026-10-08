<div align="center">

# Pretrain Once, Route Anywhere

### Towards a Foundation Model for LLM Routing

**Guannan Lai · Han-Jia Ye**<br>
School of Artificial Intelligence & National Key Laboratory for Novel Software Technology, Nanjing University

[![Paper](https://img.shields.io/badge/arXiv-2609.37362-b31b1b.svg)](https://arxiv.org/abs/2609.37362)
[![Website](https://img.shields.io/badge/Project-Website-605BF6.svg)](https://lamda-model-reuse.github.io/RouteFM/)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-D8F35A.svg)](https://huggingface.co/spaces/AIGNLAI/RouteFM-Demo)
[![PyPI](https://img.shields.io/pypi/v/routefm-router.svg)](https://pypi.org/project/routefm-router/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-RouteFM-FFD21E)](https://huggingface.co/AIGNLAI/RouteFM)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**RouteFM learns a reusable routing capability once, then adapts to new tasks,
candidate pools, and deployment conditions through behavioral context alone.**

</div>

<p align="center">
  <img src="assets/intro_new.png" width="95%" alt="RouteFM shifts LLM routing from local fitting to global routing pretraining.">
</p>

## News

- **2026-10-08:** The [project page](https://lamda-model-reuse.github.io/RouteFM/) and [interactive RouteFM-BGE demo](https://huggingface.co/spaces/AIGNLAI/RouteFM-Demo) are available.
- **2026-09-29:** The [RouteFM paper](https://arxiv.org/abs/2609.37362) is available on arXiv.
- **2026-09-29:** Code, frozen checkpoints, training configurations, and evaluation protocols are publicly released.

## Overview

Large language model routing is commonly treated as a local fitting problem:
a router is optimized for one workload and one candidate pool, then retrained
when the environment changes. RouteFM instead approaches routing as a
foundation-model problem. It learns to characterize anonymous candidate
models from a small set of behavioral observations and infer their
target-specific capabilities without relying on model identities.

After episodic pretraining across heterogeneous routing environments, the same
frozen router transfers across domains, modalities, candidate pools, and
context budgets. On MMR-Bench, which is excluded from pretraining, RouteFM
outperforms the strongest non-RouteFM baseline by **2.23 quality points** with
only eight observations per candidate.

### Highlights

- **Pretrain once:** learn a global routing prior from heterogeneous tasks and candidate pools.
- **Route anywhere:** adapt a frozen router through behavioral context, without target-domain parameter updates.
- **Identity-free candidates:** reason about anonymous models from observed quality and relative cost rather than names or provider metadata.
- **Context-efficient transfer:** deliver the largest gains when only limited behavioral evidence is available.

## Method

<p align="center">
  <img src="assets/method.png" width="95%" alt="Architecture of RouteFM.">
</p>

For each anonymous candidate, RouteFM builds a compact capability profile from
context queries and their observed quality and relative cost. A target query
retrieves complementary evidence from both the profile and the original
behavioral context. A permutation-equivariant candidate-pool Transformer then
compares the current candidates jointly and predicts target-specific quality
and relative cost.

The released architecture uses 24 capability tokens with hidden dimension
256. It is pretrained episodically with candidate-slot permutation and a
combination of point prediction, pairwise ranking, and routing-regret
objectives. See the [paper](https://arxiv.org/pdf/2609.37362) and
[pretraining guide](docs/PRETRAINING.md) for details.

## Model zoo

The canonical weights are hosted at
[AIGNLAI/RouteFM](https://huggingface.co/AIGNLAI/RouteFM). The package downloads
the safetensors file from an immutable Hub revision, verifies its SHA-256
digest, and reuses the standard Hugging Face cache.

The repository-level [`config.json`](config.json) indexes both variants and is
retrieved with each model resolution so Hugging Face can count real downloads.

| Router | Query encoder | Dim. | Query modality | Parameters | Checkpoint |
| --- | --- | ---: | --- | ---: | --- |
| RouteFM-Qwen | Qwen3-VL-Embedding-8B | 4096 | Text + image | 12.8M | [Download](https://huggingface.co/AIGNLAI/RouteFM/tree/model-v1.0.1/qwen) |
| RouteFM-BGE | BAAI/bge-base-en-v1.5 (CLS) | 768 | Text only | 11.1M | [Download](https://huggingface.co/AIGNLAI/RouteFM/tree/model-v1.0.1/bge) |

The query encoders are frozen external feature extractors and are not bundled
with RouteFM. Both routers were trained from random initialization in one
continuous 10,000-update run; they differ only in their input dimension and
encoder-specific vectors.

## Installation

RouteFM requires Python 3.10 or newer.

```bash
# Text routing with the released BGE encoder and frozen RouteFM weights.
python -m pip install "routefm-router[bge]"
```

For development or paper reproduction, install from source:

```bash
git clone https://github.com/LAMDA-Model-Reuse/RouteFM.git
cd RouteFM
python -m pip install -e '.[bge]'
```

Checkpoint prefetching:

```bash
# Download and verify both released routers for offline jobs.
routefm-download --encoder all
```

## Quick start

Use ordinary Python dictionaries to describe prior observations for each
candidate. Candidate names label the output; they are not model features.

```python
from routefm import RouteFMRouter

router = RouteFMRouter.from_pretrained(encoder="bge", device="cpu")
router.set_context({
    "small-model": [
        {"query": "What is 2 + 2?", "score": 1.0, "cost": 0.01},
        {"query": "Summarize this paragraph.", "score": 0.7, "cost": 0.02},
    ],
    "large-model": [
        {"query": "What is 2 + 2?", "score": 1.0, "cost": 0.10},
        {"query": "Summarize this paragraph.", "score": 0.95, "cost": 0.20},
    ],
})

decision = router.route("Prove that there are infinitely many primes.")
print(decision.model_name)
print(decision.predicted_scores)
```

The first call downloads the immutable RouteFM-BGE checkpoint from Hugging
Face and an immutable revision of the external BGE encoder. No RouteFM
parameters are updated.

To explore the behavioral-context interface before installing anything, use
the [live RouteFM-BGE demo](https://huggingface.co/spaces/AIGNLAI/RouteFM-Demo).
The free browser demo explores predictions precomputed with the released
checkpoint; its built-in contexts are synthetic illustrations rather than
benchmark records, and it does not invoke candidate LLMs. The full Gradio app
in [`demo/`](demo) supports arbitrary target queries and custom JSON context.

For precomputed Qwen or BGE embeddings, use `router.predict_arrays(...)` or
prepare an episode following the documented [`.npz` schema](docs/CUSTOM_DATA.md):

```bash
routefm-predict --encoder qwen --input my_episode.npz \
  --output predictions.json --device cpu
```

The file-based CLI remains available for text-only BGE embeddings:

```bash
routefm-predict --encoder bge --input my_text_episode.npz \
  --output text_predictions.json --device cpu
```

Use `--checkpoint /path/to/checkpoint.pt` for a local legacy checkpoint. A
local safetensors override must have its matching `config.json` beside it.
Standard Hugging Face settings such as `HF_HOME` and `HF_HUB_OFFLINE=1`
control the cache and offline operation.

## Evaluation and reproduction

The paper evaluates a frozen RouteFM on in-domain RouterEval tasks and on
cross-modal transfer to MMR-Bench. The official MMR-Bench result uses
dataset-wise five-fold context/target splits; RouteFM reaches **0.7323** at
K=8 versus **0.7100** for the strongest non-RouteFM baseline, and **0.7614**
in the large-observation regime.

This repository also includes an independently auditable, fixed seed-31010
MMR-Bench V1 split. It is a retrospective, post-selected illustrative split
and is deliberately documented separately from the paper's five-fold result.
After obtaining the third-party benchmark artifacts, run:

```bash
routefm-eval-small --encoder qwen --data-root /path/to/mmr_qwen \
  --output results/qwen_small.json --device cpu
routefm-eval-large --encoder qwen --data-root /path/to/mmr_qwen \
  --output results/qwen_large.json --device cpu
```

The evaluator checks the published 10,370 query IDs and nine-model order. See
[`docs/MMRBENCH_V1.md`](docs/MMRBENCH_V1.md) for the exact data contract,
split IDs, aggregation rules, reference hashes, BGE commands, and important
interpretation notes.

## Training and custom data

- [`docs/PRETRAINING.md`](docs/PRETRAINING.md): preprocessing contract, episodic sampling, curriculum, and full pretraining commands.
- [`docs/CUSTOM_DATA.md`](docs/CUSTOM_DATA.md): episode schema, embedding generation, masks, and prediction outputs.
- [`manifest.json`](manifest.json): immutable Hub revision, file sizes, configurations, and SHA-256 digests.
- [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md): provenance and terms for external datasets and encoders.

The release does not redistribute third-party datasets, benchmark images, or
embedding-model weights/services. Their respective licenses and access terms
apply separately.

## Limitations

RouteFM assumes that each candidate has valid behavioral observations and that
candidate order and embedding family remain consistent within an episode. Its
predicted cost is relative rather than a calibrated monetary or latency
estimate. The default decision rule selects maximum predicted quality and does
not impose a cost budget. Performance may change for domains, languages,
candidate pools, or encoder revisions outside the training distribution.

## Citation

If RouteFM is useful in your research, please cite:

```bibtex
@misc{lai2026pretrainoncerouteanywhere,
  title         = {Pretrain Once, Route Anywhere: Towards a Foundation Model for LLM Routing},
  author        = {Guannan Lai and Han-Jia Ye},
  year          = {2026},
  eprint        = {2609.37362},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2609.37362}
}
```

## License

RouteFM source code and released routing weights are provided under the
[Apache License 2.0](LICENSE). Third-party assets retain their original
licenses and terms.
