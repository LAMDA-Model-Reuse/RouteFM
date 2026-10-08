---
title: RouteFM Demo
emoji: 🧭
colorFrom: indigo
colorTo: green
sdk: gradio
app_file: app.py
pinned: true
license: apache-2.0
models:
  - AIGNLAI/RouteFM
---

# RouteFM interactive demo

This Space runs the released RouteFM-BGE checkpoint. It demonstrates how a
frozen, identity-free router uses a small behavioral context to select a
candidate for a new query.

- [Project page](https://lamda-model-reuse.github.io/RouteFM/)
- [Paper](https://arxiv.org/abs/2609.37362)
- [Code](https://github.com/LAMDA-Model-Reuse/RouteFM)
- [Model weights](https://huggingface.co/AIGNLAI/RouteFM)
- [Python package](https://pypi.org/project/routefm-router/)

The built-in contexts are synthetic and intended to explain the interface;
they are not benchmark samples or reported experimental results. Candidate
LLMs are not invoked. All displayed predictions come from the released frozen
router and BGE query encoder.

## Local development

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r demo/requirements.txt
python demo/app.py
```
