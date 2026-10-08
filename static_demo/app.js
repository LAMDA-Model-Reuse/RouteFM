const state = { data: null, environment: null, target: null, budget: "4", reveal: false };
const $ = (id) => document.getElementById(id);
const letter = (index) => `Candidate ${String.fromCharCode(65 + index)}`;

function fillSelect(select, values) {
  select.replaceChildren(...values.map((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    return option;
  }));
}

function candidateName(index) {
  return state.reveal ? state.data.environments[state.environment].labels[index] : letter(index);
}

function renderBudgets() {
  $("budgets").replaceChildren(...["1", "2", "4", "8"].map((value) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = `K=${value}`;
    button.className = value === state.budget ? "active" : "";
    button.addEventListener("click", () => { state.budget = value; render(); });
    return button;
  }));
}

function renderCandidates() {
  const result = state.data.environments[state.environment].results[state.budget][state.target];
  const selectedIndex = result.scores.indexOf(Math.max(...result.scores));
  $("choice").textContent = candidateName(selectedIndex);
  $("context-summary").textContent = `K=${state.budget} observations per candidate`;
  $("candidate-grid").replaceChildren(...result.scores.map((score, index) => {
    const card = document.createElement("article");
    card.className = `candidate${index === selectedIndex ? " selected" : ""}`;
    const name = document.createElement("div");
    name.className = "candidate-top";
    const strong = document.createElement("strong");
    strong.textContent = candidateName(index);
    name.append(strong);
    if (index === selectedIndex) {
      const badge = document.createElement("span"); badge.textContent = "selected"; name.append(badge);
    }
    card.append(name, measure("Predicted quality", score, false), measure("Predicted relative cost", result.costs[index], true));
    return card;
  }));
}

function measure(label, value, cost) {
  const wrap = document.createElement("div"); wrap.className = `measure${cost ? " cost" : ""}`;
  const row = document.createElement("label"); row.textContent = label;
  const number = document.createElement("b"); number.textContent = value.toFixed(3); row.append(number);
  const bar = document.createElement("div"); bar.className = "bar";
  const fill = document.createElement("i"); fill.style.width = `${Math.min(100, Math.max(0, value * 100))}%`; bar.append(fill);
  wrap.append(row, bar); return wrap;
}

function renderContext() {
  const env = state.data.environments[state.environment];
  const header = document.createElement("tr");
  ["Context query", ...env.labels.flatMap((_, index) => [`${String.fromCharCode(65 + index)} · score`, `${String.fromCharCode(65 + index)} · cost`])].forEach((value) => {
    const th = document.createElement("th"); th.textContent = value; header.append(th);
  });
  $("context-head").replaceChildren(header);
  $("context-body").replaceChildren(...env.observations.slice(0, Number(state.budget)).map((observation) => {
    const tr = document.createElement("tr");
    [observation.query, ...observation.scores.flatMap((score, index) => [score.toFixed(2), observation.costs[index].toFixed(2)])].forEach((value) => {
      const td = document.createElement("td"); td.textContent = value; tr.append(td);
    });
    return tr;
  }));
}

function render() { renderBudgets(); renderCandidates(); renderContext(); }

fetch("data.json").then((response) => {
  if (!response.ok) throw new Error(`Could not load demo data (${response.status}).`);
  return response.json();
}).then((data) => {
  state.data = data;
  const environments = Object.keys(data.environments);
  state.environment = environments[0];
  fillSelect($("environment"), environments);
  fillSelect($("target"), data.environments[state.environment].targets);
  state.target = data.environments[state.environment].targets[0];
  $("environment").addEventListener("change", (event) => {
    state.environment = event.target.value;
    const targets = data.environments[state.environment].targets;
    fillSelect($("target"), targets); state.target = targets[0]; render();
  });
  $("target").addEventListener("change", (event) => { state.target = event.target.value; renderCandidates(); });
  $("reveal").addEventListener("change", (event) => { state.reveal = event.target.checked; renderCandidates(); });
  render();
}).catch((error) => {
  $("candidate-grid").textContent = error.message;
});
