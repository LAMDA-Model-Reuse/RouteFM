"""Interactive Hugging Face Space for the released RouteFM-BGE router."""
from __future__ import annotations

from functools import lru_cache
import html
import json
import threading
from typing import Any

import gradio as gr

from routefm import RouteFMRouter


MAX_CUSTOM_CHARACTERS = 20_000
MAX_CANDIDATES = 8
MAX_OBSERVATIONS = 16
CANDIDATE_LETTERS = "ABCDEFGH"
MODEL_LOCK = threading.Lock()


ENVIRONMENTS: dict[str, dict[str, Any]] = {
    "Mixed assistant": {
        "query": "Prove that there are infinitely many prime numbers.",
        "labels": [
            "Fast generalist",
            "Reasoning specialist",
            "Code specialist",
            "Broad premium model",
        ],
        "observations": [
            ("Summarize the causes of the 2008 financial crisis.", [0.76, 0.80, 0.62, 0.91], [0.08, 0.38, 0.22, 0.92]),
            ("Solve: if 3x + 7 = 25, find x.", [0.88, 0.98, 0.70, 0.98], [0.06, 0.34, 0.20, 0.88]),
            ("Write a Python function for binary search.", [0.71, 0.75, 0.98, 0.94], [0.09, 0.40, 0.24, 0.95]),
            ("Explain why the sky appears blue.", [0.84, 0.89, 0.72, 0.95], [0.07, 0.36, 0.21, 0.90]),
            ("Find the derivative of x^3 sin(x).", [0.72, 0.97, 0.78, 0.98], [0.07, 0.36, 0.22, 0.91]),
            ("Debug a race condition in a producer-consumer queue.", [0.52, 0.70, 0.96, 0.93], [0.10, 0.44, 0.27, 1.00]),
            ("Draft a concise, polite meeting follow-up email.", [0.91, 0.84, 0.69, 0.94], [0.06, 0.35, 0.20, 0.87]),
            ("Compare utilitarian and deontological ethics.", [0.74, 0.86, 0.60, 0.95], [0.08, 0.40, 0.23, 0.96]),
        ],
    },
    "Reasoning lab": {
        "query": "A fair coin is tossed until two consecutive heads appear. What is the expected number of tosses?",
        "labels": [
            "Compact solver",
            "Symbolic reasoner",
            "Fast chat model",
            "Large reasoner",
        ],
        "observations": [
            ("Prove that sqrt(2) is irrational.", [0.63, 0.96, 0.51, 0.98], [0.09, 0.46, 0.05, 1.00]),
            ("Compute the eigenvalues of [[2,1],[1,2]].", [0.78, 0.97, 0.59, 0.97], [0.08, 0.43, 0.05, 0.96]),
            ("Solve the recurrence a_n = 2a_(n-1) + 1.", [0.76, 0.95, 0.56, 0.98], [0.08, 0.42, 0.05, 0.98]),
            ("Is every continuous function differentiable? Explain.", [0.89, 0.94, 0.88, 0.96], [0.07, 0.39, 0.04, 0.92]),
            ("Derive Bayes' rule from conditional probability.", [0.81, 0.96, 0.68, 0.98], [0.09, 0.45, 0.05, 1.00]),
            ("Find a counterexample to the converse of Lagrange's theorem.", [0.48, 0.91, 0.34, 0.95], [0.10, 0.49, 0.06, 1.05]),
            ("Calculate 17 × 24.", [0.99, 1.00, 0.98, 1.00], [0.04, 0.27, 0.03, 0.70]),
            ("Explain the intuition behind the central limit theorem.", [0.75, 0.93, 0.73, 0.97], [0.08, 0.43, 0.05, 0.97]),
        ],
    },
    "Software team": {
        "query": "Design an idempotent webhook consumer with retries and exactly-once effects.",
        "labels": [
            "Low-latency coder",
            "Systems specialist",
            "General assistant",
            "Advanced coding model",
        ],
        "observations": [
            ("Implement merge sort in Python.", [0.94, 0.96, 0.79, 0.99], [0.08, 0.38, 0.06, 0.90]),
            ("Explain the difference between a process and a thread.", [0.82, 0.96, 0.91, 0.97], [0.07, 0.37, 0.05, 0.88]),
            ("Find the memory leak in this ownership graph.", [0.62, 0.94, 0.55, 0.98], [0.10, 0.45, 0.07, 1.00]),
            ("Write a SQL query with a window function for running totals.", [0.86, 0.97, 0.76, 0.98], [0.08, 0.41, 0.06, 0.94]),
            ("Review an API design for backwards compatibility.", [0.70, 0.96, 0.81, 0.98], [0.09, 0.44, 0.06, 1.00]),
            ("Fix an off-by-one error in binary search.", [0.96, 0.97, 0.83, 0.99], [0.06, 0.34, 0.05, 0.83]),
            ("Explain eventual consistency to a product manager.", [0.73, 0.92, 0.95, 0.96], [0.07, 0.38, 0.05, 0.90]),
            ("Design a rate limiter for a distributed API gateway.", [0.58, 0.97, 0.66, 0.98], [0.11, 0.49, 0.07, 1.04]),
        ],
    },
}


CUSTOM_EXAMPLE = json.dumps(
    {
        "small-model": [
            {"query": "What is 2 + 2?", "score": 1.0, "cost": 0.01},
            {"query": "Summarize a news article.", "score": 0.72, "cost": 0.02},
        ],
        "large-model": [
            {"query": "What is 2 + 2?", "score": 1.0, "cost": 0.10},
            {"query": "Summarize a news article.", "score": 0.95, "cost": 0.20},
        ],
    },
    indent=2,
)


@lru_cache(maxsize=1)
def get_router() -> RouteFMRouter:
    """Load both frozen components once per Space process."""
    return RouteFMRouter.from_pretrained(
        encoder="bge",
        device="cpu",
        embedding_batch_size=32,
        embedding_max_length=256,
    )


def display_names(labels: list[str], reveal: bool) -> list[str]:
    if reveal:
        return labels
    return [f"Candidate {CANDIDATE_LETTERS[index]}" for index in range(len(labels))]


def preset_context(environment: str, budget: int, reveal: bool) -> dict[str, list[dict[str, Any]]]:
    item = ENVIRONMENTS[environment]
    labels = display_names(item["labels"], reveal)
    observations = item["observations"][: int(budget)]
    return {
        label: [
            {"query": query, "score": scores[index], "cost": costs[index]}
            for query, scores, costs in observations
        ]
        for index, label in enumerate(labels)
    }


def validate_custom_context(raw: str, reveal: bool) -> dict[str, list[dict[str, Any]]]:
    if len(raw) > MAX_CUSTOM_CHARACTERS:
        raise gr.Error(f"Custom JSON must be under {MAX_CUSTOM_CHARACTERS:,} characters.")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise gr.Error(f"Invalid JSON: {error.msg} at line {error.lineno}.") from error
    if not isinstance(value, dict) or not 2 <= len(value) <= MAX_CANDIDATES:
        raise gr.Error(f"Provide an object containing 2–{MAX_CANDIDATES} candidates.")
    labels = list(value)
    if any(not isinstance(label, str) or not label.strip() for label in labels):
        raise gr.Error("Every candidate name must be a non-empty string.")
    shown = display_names(labels, reveal)
    context: dict[str, list[dict[str, Any]]] = {}
    for candidate_index, (label, rows) in enumerate(value.items()):
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_OBSERVATIONS:
            raise gr.Error(
                f"{label!r} must have 1–{MAX_OBSERVATIONS} observations."
            )
        clean_rows = []
        for row_index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                raise gr.Error(f"Observation {row_index} for {label!r} must be an object.")
            try:
                query = row["query"]
                score = float(row["score"])
                cost = float(row["cost"])
            except (KeyError, TypeError, ValueError) as error:
                raise gr.Error(
                    f"Observation {row_index} for {label!r} needs query, numeric score, and numeric cost."
                ) from error
            if not isinstance(query, str) or not query.strip():
                raise gr.Error(f"Observation {row_index} for {label!r} has an empty query.")
            if not 0.0 <= score <= 1.0 or cost < 0.0:
                raise gr.Error("Scores must be in [0, 1] and costs must be nonnegative.")
            clean_rows.append({"query": query.strip(), "score": score, "cost": cost})
        context[shown[candidate_index]] = clean_rows
    return context


def context_rows(environment: str, budget: int) -> list[list[Any]]:
    rows = []
    for query, scores, costs in ENVIRONMENTS[environment]["observations"][: int(budget)]:
        rows.append(
            [query]
            + [value for pair in zip(scores, costs, strict=True) for value in pair]
        )
    return rows


def select_environment(environment: str, budget: int) -> tuple[str, list[list[Any]]]:
    return ENVIRONMENTS[environment]["query"], context_rows(environment, budget)


def update_context_table(environment: str, budget: int) -> list[list[Any]]:
    return context_rows(environment, budget)


def render_result(decision: Any, context_size: int) -> str:
    scores = decision.predicted_scores
    costs = decision.predicted_relative_costs
    cards = []
    for name, score in scores.items():
        safe_name = html.escape(name)
        safe_score = min(max(float(score), 0.0), 1.0)
        safe_cost = min(max(float(costs[name]), 0.0), 1.0)
        selected = name == decision.model_name
        cards.append(
            f"""
            <div class="rf-candidate {'rf-selected' if selected else ''}">
              <div class="rf-candidate-top">
                <strong>{safe_name}</strong>
                {'<span>selected</span>' if selected else ''}
              </div>
              <div class="rf-measure"><label>Predicted quality <b>{safe_score:.3f}</b></label><i><em style="width:{safe_score * 100:.1f}%"></em></i></div>
              <div class="rf-measure rf-cost"><label>Predicted relative cost <b>{safe_cost:.3f}</b></label><i><em style="width:{safe_cost * 100:.1f}%"></em></i></div>
            </div>
            """
        )
    return f"""
    <div class="rf-result">
      <div class="rf-decision">
        <div><small>ROUTED TO</small><h2>{html.escape(decision.model_name)}</h2></div>
        <p>Highest predicted quality<br><span>{context_size} observations per candidate</span></p>
      </div>
      <div class="rf-candidates">{''.join(cards)}</div>
      <p class="rf-rule">RouteFM predicts quality and relative cost jointly. The released default decision rule selects the highest predicted quality; cost is displayed but not used for selection.</p>
    </div>
    """


def route_query(
    environment: str,
    budget: int,
    query: str,
    reveal: bool,
    use_custom: bool,
    custom_json: str,
) -> str:
    if not isinstance(query, str) or not query.strip():
        raise gr.Error("Enter a non-empty target query.")
    if len(query) > 4_000:
        raise gr.Error("Target query must be under 4,000 characters.")
    context = (
        validate_custom_context(custom_json, reveal)
        if use_custom
        else preset_context(environment, budget, reveal)
    )
    context_size = min(len(rows) for rows in context.values())
    try:
        # RouteFMRouter stores the active context, so protect the set-and-route
        # pair when a public Space handles concurrent requests.
        with MODEL_LOCK:
            decision = get_router().set_context(context).route(query.strip())
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(
            "RouteFM could not complete this request. The first request may still be "
            "downloading the frozen model; retry once, or inspect the Space logs. "
            f"Technical detail: {type(error).__name__}: {error}"
        ) from error
    return render_result(decision, context_size)


TABLE_HEADERS = ["Context query"] + [
    value
    for letter in CANDIDATE_LETTERS[:4]
    for value in (f"{letter} · score", f"{letter} · cost")
]

SPACE_CSS = """
.gradio-container { max-width: 1180px !important; font-family: Inter, ui-sans-serif, system-ui, sans-serif !important; }
.rf-hero { border: 1px solid #dfe3eb; border-radius: 12px; padding: 24px 28px; background: linear-gradient(135deg,#081424,#101e35); color: white; margin-bottom: 16px; }
.rf-hero small { color:#d8f35a; font: 11px ui-monospace,monospace; letter-spacing:.12em; }
.rf-hero h1 { margin:8px 0 5px; font-size:34px; letter-spacing:-.04em; }
.rf-hero p { margin:0; color:#aeb9c9; max-width:760px; }
.rf-note { font-size:12px; color:#667085; }
.rf-result { border:1px solid #dfe3eb; border-radius:12px; overflow:hidden; background:#fff; }
.rf-decision { display:flex; justify-content:space-between; gap:20px; align-items:center; padding:20px 22px; color:#fff; background:#0b1729; }
.rf-decision small { color:#d8f35a; font:10px ui-monospace,monospace; letter-spacing:.14em; }
.rf-decision h2 { margin:3px 0 0; color:#fff; font-size:25px; }
.rf-decision p { margin:0; text-align:right; font-size:12px; color:#aab6c7; }.rf-decision p span { color:#d8f35a; }
.rf-candidates { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:1px; background:#e5e7eb; }
.rf-candidate { padding:18px; background:#fff; }.rf-candidate.rf-selected { background:#f7f7ff; box-shadow:inset 3px 0 #605bf6; }
.rf-candidate-top { display:flex; justify-content:space-between; gap:10px; margin-bottom:15px; }.rf-candidate-top strong { font-size:13px; }.rf-candidate-top span { color:#443edc; background:#ecebff; border-radius:20px; padding:2px 7px; font-size:9px; text-transform:uppercase; }
.rf-measure { margin:10px 0; }.rf-measure label { display:flex; justify-content:space-between; color:#677287; font-size:10px; }.rf-measure label b { color:#1f2937; }
.rf-measure i { display:block; height:5px; margin-top:5px; background:#edf0f4; }.rf-measure em { display:block; height:100%; background:#605bf6; }.rf-cost em { background:#40bba9; }
.rf-rule { margin:0; padding:13px 18px; border-top:1px solid #e5e7eb; color:#697386; font-size:10px; }
@media(max-width:650px){.rf-candidates{grid-template-columns:1fr}.rf-decision{align-items:flex-start;flex-direction:column}.rf-decision p{text-align:left}}
"""


with gr.Blocks(css=SPACE_CSS, title="RouteFM Demo") as demo:
    gr.HTML(
        """
        <div class="rf-hero">
          <small>INTERACTIVE ROUTING LAB</small>
          <h1>Pretrain once. Route anywhere.</h1>
          <p>Show a frozen RouteFM router how anonymous candidates behaved, then ask where a new query should go.</p>
        </div>
        """
    )
    with gr.Row():
        with gr.Column(scale=5):
            environment = gr.Dropdown(
                choices=list(ENVIRONMENTS),
                value="Mixed assistant",
                label="1 · Choose an illustrative environment",
            )
        with gr.Column(scale=3):
            context_budget = gr.Radio(
                choices=[1, 2, 4, 8],
                value=4,
                label="2 · Observations per candidate (K)",
            )
        with gr.Column(scale=3):
            reveal_labels = gr.Checkbox(
                value=False,
                label="Reveal descriptive labels",
                info="Names label results; RouteFM never sees them as features.",
            )

    target_query = gr.Textbox(
        value=ENVIRONMENTS["Mixed assistant"]["query"],
        label="3 · Enter a target query",
        lines=3,
        max_lines=8,
    )
    run_button = gr.Button("Route this query", variant="primary")
    result = gr.HTML(
        "<div class='rf-note'>Run the router to compare predicted quality and relative cost across candidates. The first request downloads and loads the frozen weights.</div>"
    )

    with gr.Accordion("Inspect the behavioral context", open=False):
        gr.Markdown(
            "Each row is a synthetic observation available for every candidate. "
            "Scores are in **[0, 1]**; costs are illustrative relative values."
        )
        context_table = gr.Dataframe(
            headers=TABLE_HEADERS,
            value=context_rows("Mixed assistant", 4),
            datatype=["str"] + ["number"] * 8,
            interactive=False,
            wrap=True,
        )

    with gr.Accordion("Advanced · Route with your own context", open=False):
        use_custom = gr.Checkbox(
            value=False,
            label="Use the JSON below instead of the selected preset",
        )
        custom_context = gr.Code(
            value=CUSTOM_EXAMPLE,
            language="json",
            label="Candidate → observations",
        )
        gr.Markdown(
            "Provide 2–8 candidates and 1–16 observations per candidate. Candidate "
            "names are output labels only. Do not paste private or sensitive prompts into a public Space."
        )

    gr.HTML(
        """
        <p class="rf-note">This demonstration uses the released <b>RouteFM-BGE</b> checkpoint and its frozen BGE query encoder. Presets are illustrative, not benchmark examples. No candidate LLM is called and no answer text is generated. <a href="https://arxiv.org/abs/2609.37362" target="_blank">Paper ↗</a> · <a href="https://github.com/LAMDA-Model-Reuse/RouteFM" target="_blank">Code ↗</a></p>
        """
    )

    environment.change(
        fn=select_environment,
        inputs=[environment, context_budget],
        outputs=[target_query, context_table],
    )
    context_budget.change(
        fn=update_context_table,
        inputs=[environment, context_budget],
        outputs=context_table,
    )
    run_button.click(
        fn=route_query,
        inputs=[
            environment,
            context_budget,
            target_query,
            reveal_labels,
            use_custom,
            custom_context,
        ],
        outputs=result,
    )
    target_query.submit(
        fn=route_query,
        inputs=[
            environment,
            context_budget,
            target_query,
            reveal_labels,
            use_custom,
            custom_context,
        ],
        outputs=result,
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
