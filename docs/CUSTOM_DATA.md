# Route custom data

RouteFM takes observed Context per candidate and target-query embeddings.
It does not use a candidate's name as a feature. Any candidate can be added
without training router parameters once that candidate has at least one
observed Context outcome. Context scores should use a comparable [0,1]
quality scale across candidates; costs must be nonnegative and use one
consistent unit within the episode.

Store a NumPy `.npz` archive with these arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| `context_embeddings` | `[M,K,D]` | Historical query vector for each candidate |
| `context_scores` | `[M,K]` | Observed candidate quality |
| `context_costs` | `[M,K]` | Observed cost |
| `context_mask` | `[M,K]` | Optional Boolean validity; padding is false |
| `target_embeddings` | `[T,D]` | New queries to route |

Here D=4096 for Qwen or D=768 for BGE. The first axis has the same candidate
ordering in every Context array. Different candidates may have different
numbers of observed queries: pad to a common K and set false in the mask.
Every candidate must have at least one true mask cell. No Target score, cost,
or answer is supplied. The same query can be observed for multiple models,
but a Target query must not be copied into Context when evaluating held-out
performance. For ordinary deployment, Context is prior observed traffic.

Illustrative packaging code, after you have obtained the embeddings and
observations:

```python
import numpy as np

# Replace these placeholders with your actual M candidates and T targets.
# Each context row may contain a different query set.
np.savez(
    "my_episode.npz",
    context_embeddings=context_vectors,  # float32 [M,K,D]
    context_scores=observed_scores,       # float32 [M,K], [0,1]
    context_costs=observed_costs,         # float32 [M,K], >=0
    context_mask=observed_mask,           # bool [M,K]
    target_embeddings=target_vectors,     # float32 [T,D]
)
```

The Qwen router requires **Qwen3-VL-Embedding-8B** 4096-D L2-normalized
vectors. For an image query, embed its question and image *jointly* with a
compatible provider; embedding only text loses the image signal. A JSONL file
with one provider-compatible `input` value per line can be sent via:

```bash
export ROUTEFM_EMBEDDING_API_KEY='YOUR_KEY'
routefm-embed-qwen --input-jsonl queries.jsonl --output vectors.npy \
  --base-url YOUR_EMBEDDING_BASE_URL --model YOUR_QWEN3_VL_EMBEDDING_MODEL
```

The BGE router requires **BAAI/bge-base-en-v1.5**, CLS pooling and L2
normalization. It is text-only; create JSONL rows with a `query_text` field:

```bash
routefm-embed-bge --input-jsonl queries.jsonl --output vectors.npy \
  --model BAAI/bge-base-en-v1.5 --device cpu
```

Use the same encoder and revision for both Context and Target. Do not project
one encoder's vectors into another dimension or mix the two. Then route:

```bash
routefm-predict --encoder qwen --input my_episode.npz \
  --model-names candidate_names.json --output predictions.json --device cpu
```

`candidate_names.json` is an optional JSON array of M names in Context row
order. `chosen_model_index[t]` gives the selected candidate for Target t;
`predicted_score[t][m]` is its estimated quality. The default rule is highest
predicted quality. `predicted_relative_cost` is an auxiliary output, not
calibrated dollars or latency, and is not used for this selection. No router
update is performed. The `.npz` example is an inference episode, not a
pretraining dataset; see [`PRETRAINING.md`](PRETRAINING.md) for the latter.
