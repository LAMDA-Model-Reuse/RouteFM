---
title: RouteFM Demo
emoji: 🧭
colorFrom: indigo
colorTo: green
sdk: static
app_file: index.html
pinned: true
license: apache-2.0
models:
  - AIGNLAI/RouteFM
tags:
  - arxiv:2609.37362
short_description: Explore identity-free RouteFM routing across contexts.
---

# RouteFM interactive result explorer

This free static Space visualizes predictions precomputed with the released
RouteFM-BGE checkpoint. Change the environment, target query, and context
budget to inspect how the frozen router compares anonymous candidates.

The observations are synthetic illustrations rather than benchmark records.
For arbitrary queries and custom candidate contexts, install
`routefm-router[bge]` or run the full Gradio application from the
[GitHub repository](https://github.com/LAMDA-Model-Reuse/RouteFM/tree/main/demo).

- [Project page](https://lamda-model-reuse.github.io/RouteFM/)
- [Paper](https://arxiv.org/abs/2609.37362)
- [Code](https://github.com/LAMDA-Model-Reuse/RouteFM)
- [Model weights](https://huggingface.co/AIGNLAI/RouteFM)
- [LLM routing research collection](https://huggingface.co/collections/AIGNLAI/llm-routing-research-guannan-lai-6ac753db0a06781003fa245d)
