# RouteFM

[![Paper](https://img.shields.io/badge/arXiv-2609.37362-b31b1b.svg)](https://arxiv.org/abs/2609.37362)
[![Website](https://img.shields.io/badge/Project-Website-605BF6.svg)](https://lamda-model-reuse.github.io/RouteFM/)
[![Demo](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-D8F35A.svg)](https://huggingface.co/spaces/AIGNLAI/RouteFM-Demo)
[![Colab](https://img.shields.io/badge/Colab-Run%20RouteFM-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/LAMDA-Model-Reuse/RouteFM/blob/main/examples/routefm_colab.ipynb)
[![Models](https://img.shields.io/badge/Hugging%20Face-RouteFM-FFD21E.svg)](https://huggingface.co/AIGNLAI/RouteFM)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/LAMDA-Model-Reuse/RouteFM/blob/main/LICENSE)

Official Python package for **Pretrain Once, Route Anywhere: Towards a
Foundation Model for LLM Routing**.

RouteFM is a pretrained router that adapts a frozen model to new tasks and
candidate pools through behavioral Context. Candidate identities are not used
as features: each candidate is characterized by a small number of observed
query, quality, and relative-cost tuples.

![RouteFM overview](https://raw.githubusercontent.com/LAMDA-Model-Reuse/RouteFM/main/assets/intro_new.png)

## Install

RouteFM requires Python 3.10 or newer. Install the BGE text interface with:

```bash
python -m pip install "routefm-router[bge]"
```

The RouteFM package is lightweight; released routing checkpoints remain on
[Hugging Face](https://huggingface.co/AIGNLAI/RouteFM) and are downloaded and
checksum-verified on first use.
The built-in text API also pins the compatible BGE encoder revision.

## Route text in Python

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

The default decision rule selects the highest predicted quality. Predicted
cost is relative rather than calibrated currency or latency and is not used
by that rule. No RouteFM parameter update occurs.

For precomputed BGE or Qwen multimodal embeddings, use
`RouteFMRouter.predict_arrays(...)`. The existing `routefm-predict` CLI and
`.npz` workflow remain supported.

## Resources

- [Project page](https://lamda-model-reuse.github.io/RouteFM/)
- [Interactive RouteFM-BGE demo](https://huggingface.co/spaces/AIGNLAI/RouteFM-Demo)
- [One-click RouteFM-BGE Colab](https://colab.research.google.com/github/LAMDA-Model-Reuse/RouteFM/blob/main/examples/routefm_colab.ipynb)
- [Paper](https://arxiv.org/abs/2609.37362)
- [Source and documentation](https://github.com/LAMDA-Model-Reuse/RouteFM)
- [Frozen checkpoints and model card](https://huggingface.co/AIGNLAI/RouteFM)
- [Custom-data schema](https://github.com/LAMDA-Model-Reuse/RouteFM/blob/main/docs/CUSTOM_DATA.md)
- [Pretraining guide](https://github.com/LAMDA-Model-Reuse/RouteFM/blob/main/docs/PRETRAINING.md)

## Citation

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

RouteFM source code and released routing weights use the Apache License 2.0.
Third-party encoders and datasets retain their own licenses and terms.
