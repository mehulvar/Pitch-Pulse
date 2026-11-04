// The model card at the bottom of the page: what each model was trained on and how it scored
// on seasons it never saw. Uses getJson from the main script on the page.

const SEASON = s => `${s}/${String((s + 1) % 100).padStart(2, "0")}`
const seasonList = seasons => seasons.map(SEASON).join(" and ")
const withRange = (ci, digits) => `${ci.estimate.toFixed(digits)} [${ci.low.toFixed(digits)}, ${ci.high.toFixed(digits)}]`

// Small helpers that build elements with plain text, never HTML strings
function el(tag, text, className) {
  const node = document.createElement(tag)
  if (text !== undefined) node.textContent = text
  if (className) node.className = className
  return node
}

function table(headers, rows) {
  const t = el("table", undefined, "mc-table")
  const head = el("tr")
  headers.forEach(h => head.append(el("th", h)))
  t.append(head)
  for (const row of rows) {
    const tr = el("tr")
    row.forEach((cell, i) => tr.append(el("td", cell, i > 0 ? "num" : undefined)))
    t.append(tr)
  }
  return t
}

function xgSection(m) {
  const h = m.holdout
  const nodes = [
    el("h3", "Expected goals (xG) model"),
    el("p", `LightGBM classifier, settings tuned with Optuna on ${SEASON(m.dev_seasons[0])} to ${SEASON(m.dev_seasons.at(-1))}` +
      `${m.monotone_constraints ? ", with monotonic constraints on distance and angle" : ""}. ` +
      `Tested on ${seasonList(m.holdout_seasons)}: ${h.shots.toLocaleString()} shots, ${h.goals} goals ` +
      `(our total xG ${h.total_xg_ours}, Understat's ${h.total_xg_understat}).`),
    table(["", "Brier score", "Log loss"], [
      ["Our model", withRange(h.brier_ours, 4), withRange(h.log_loss_ours, 4)],
      ["Understat xG", withRange(h.brier_understat, 4), withRange(h.log_loss_understat, 4)],
      ["Ours minus Understat", withRange(h.brier_gap_to_understat, 4), withRange(h.log_loss_gap_to_understat, 4)],
    ]),
    el("p", "Calibration: each dot is a tenth of the test shots. On the diagonal means the predicted chance matched how often those shots really went in."),
  ]
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg")
  svg.id = "calibration"
  nodes.push(svg)
  return { nodes, draw: () => drawCalibration(svg, h.calibration) }
}

function drawCalibration(node, rows) {
  const size = 260, pad = 36
  const top = Math.max(...rows.map(r => Math.max(r.predicted, r.actual))) * 1.05
  const svg = d3.select(node).attr("viewBox", `0 0 ${size} ${size}`)
  const x = d3.scaleLinear([0, top], [pad, size - 10])
  const y = d3.scaleLinear([0, top], [size - pad, 10])
  svg.append("g").attr("transform", `translate(0,${size - pad})`).call(d3.axisBottom(x).ticks(4, "%"))
  svg.append("g").attr("transform", `translate(${pad},0)`).call(d3.axisLeft(y).ticks(4, "%"))
  svg.append("line").attr("x1", x(0)).attr("y1", y(0)).attr("x2", x(top)).attr("y2", y(top))
    .attr("stroke", "#999").attr("stroke-dasharray", "4 3")
  svg.selectAll("circle").data(rows).join("circle")
    .attr("cx", r => x(r.predicted)).attr("cy", r => y(r.actual)).attr("r", 4).attr("fill", "#1565c0")
  svg.append("text").attr("x", size / 2).attr("y", size - 4).attr("text-anchor", "middle")
    .attr("font-size", 10).text("predicted chance")
  svg.append("text").attr("transform", `translate(10,${size / 2}) rotate(-90)`).attr("text-anchor", "middle")
    .attr("font-size", 10).text("actual goal rate")
}

function shotsSection(m) {
  const share = p => `${Math.round(p * 100)}%`
  const rows = Object.entries(m.holdout).map(([name, r]) => [
    name + (name === m.winner ? " (used)" : ""), withRange(r.avg_miss, 2),
    share(r.interval_80_coverage_poisson), share(r.interval_80_coverage_negative_binomial),
  ])
  return { nodes: [
    el("h3", "Shots-conceded model"),
    el("p", `Two candidates tuned on earlier seasons; ${m.winner} won on practice rounds. ` +
      `Tested on ${seasonList(m.holdout_seasons)} (${m.holdout_team_matches.toLocaleString()} team-matches).`),
    table(["", "Average miss (shots)", "In 80% range: Poisson", "In 80% range: negative binomial"], rows),
    el("p", "An honest 80% range should hold the real count at least 80% of the time. The Poisson range falls short " +
      "because shot counts vary more than it assumes, so the page uses the negative binomial range, " +
      "with the extra spread measured on the training seasons only."),
  ] }
}

function matchSection(m) {
  const rows = Object.entries(m.scores).map(([name, s]) => [
    name, withRange(s.rps, 4), withRange(s.log_loss, 3), withRange(s.rps_minus_market, 4),
  ])
  const c = m.chosen
  return { nodes: [
    el("h3", "Match outcome model"),
    el("p", `Bayesian hierarchical model (PyMC) learning from ${c.target === "xg" ? "xG" : "goals"}, ` +
      `with a ${c.half_life_days}-day half-life, both chosen on ${SEASON(m.tuning_season)}. ` +
      `Backtested on ${seasonList(m.holdout_seasons)} (${m.holdout_matches} matches), refitted every week on earlier matches only.`),
    table(["", "Ranked probability score", "Log loss", "RPS minus bookmakers"], rows),
    el("p", "Lower is better for every score. A negative last column would mean beating the bookmakers' closing odds, which very few public models manage. " +
      "The hindsight row is not a forecast: Understat works it out after the match from that match's own chances, " +
      "so it shows how much of a result even perfect knowledge of the chances can explain. The rest is finishing, goalkeeping and luck."),
  ] }
}

const LIMITS = "What the models don't know: defender and goalkeeper positions, shot power, player quality, injuries and line-ups. Premier League only."

async function renderModelCard() {
  const box = document.getElementById("model-card")
  try {
    const m = await getJson("/api/model-card")
    const sections = [
      m.xg_model && xgSection(m.xg_model),
      m.shots_model && shotsSection(m.shots_model),
      m.match_model && matchSection(m.match_model),
    ].filter(Boolean)
    if (!sections.length) {
      box.replaceChildren(el("p", "No training results yet. Run the training scripts to fill this in."))
      return
    }
    box.replaceChildren(...sections.flatMap(s => s.nodes), el("p", LIMITS))
    sections.forEach(s => s.draw && s.draw())
  } catch (err) {
    box.replaceChildren(el("p", `Couldn't load the model card: ${err.message}`))
  }
}

renderModelCard()
