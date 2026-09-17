import { useEffect, useMemo, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

const presets = [
  { id: "fast",     name: "Fast",     note: "Quick runs, lower complexity",           config: { train_end_year: 2015, n_estimators: 150, max_depth: 4,  class_weight: "balanced", random_state: 42 } },
  { id: "balanced", name: "Balanced", note: "Solid default",                          config: { train_end_year: 2010, n_estimators: 300, max_depth: 6,  class_weight: "balanced", random_state: 42 } },
  { id: "deep",     name: "Deep",     note: "More capacity, slower to train",         config: { train_end_year: 2000, n_estimators: 500, max_depth: 10, class_weight: "balanced", random_state: 42 } },
];

const pages = [
  { id: "predict",   label: "Predict" },
  { id: "breakdown", label: "Race Breakdown" },
  { id: "forecast",  label: "Forecast" },
  { id: "overview",  label: "Overview" },
  { id: "saved",     label: "Saved" },
  { id: "metrics",   label: "About Metrics" },
];

const teamColors = {
  "Red Bull": "#1E41FF", "Red Bull Racing": "#1E41FF",
  Ferrari: "#DC0000", Mercedes: "#00D2BE", McLaren: "#FF8700",
  "Aston Martin": "#006F62", "Alpine F1 Team": "#0090FF", Alpine: "#0090FF",
  Williams: "#005AFF", Sauber: "#00FF87", "Alfa Romeo": "#900000",
  AlphaTauri: "#2B4562", RB: "#2B4562", "Haas F1 Team": "#B6BABD", Haas: "#B6BABD",
  "Racing Point": "#F596C8", Renault: "#FFD800", "Toro Rosso": "#469BFF",
  "Force India": "#F596C8", Jordan: "#FFB400", Benetton: "#00A19B",
  Lotus: "#006F62", "Lotus F1 Team": "#111111", Brawn: "#D8FF00",
  BAR: "#E10600", Jaguar: "#008B5B",
};

const teamAbbreviations = {
  "Red Bull": "RBR", "Red Bull Racing": "RBR",
  Ferrari: "FER", Mercedes: "AMG", McLaren: "MCL",
  "Aston Martin": "AMR", "Alpine F1 Team": "ALP", Alpine: "ALP",
  Williams: "WIL", Sauber: "SAU", "Alfa Romeo": "ALF",
  AlphaTauri: "APT", RB: "RB", "RB F1 Team": "RB",
  "Haas F1 Team": "HAA", Haas: "HAA",
  "Racing Point": "RAP", Renault: "REN", "Toro Rosso": "STR",
  "Force India": "FIN", Jordan: "JOR", Benetton: "BEN",
  Lotus: "LOT", "Lotus F1 Team": "LOT", Brawn: "BGP",
  BAR: "BAR", Jaguar: "JAG", "Cadillac F1 Team": "CAD", Audi: "AUD",
};

const teamShortNames = {
  "Red Bull": "Red Bull", "Red Bull Racing": "Red Bull",
  "Haas F1 Team": "Haas", "Alpine F1 Team": "Alpine",
  "Aston Martin": "Aston Martin", "Alfa Romeo": "Alfa Romeo",
  "Lotus F1 Team": "Lotus", "Force India": "Force India",
  "Racing Point": "Racing Point", "Toro Rosso": "Toro Rosso",
  "RB F1 Team": "RB", "Cadillac F1 Team": "Cadillac",
};
const shortName = (name) => teamShortNames[name] || name.split(" ")[0];

function TeamLogo({ name, color, size = 22 }) {
  const abbr = teamAbbreviations[name] || name.slice(0, 3).toUpperCase();
  const textColor = ["#FFD800", "#D8FF00", "#00FF87", "#B6BABD", "#FF8700"].includes(color) ? "#1b1b1f" : "#ffffff";
  return (
    <span
      className="team-badge"
      style={{ background: color, width: size, height: size, fontSize: Math.round(size * 0.37), color: textColor }}
      title={name}
    >
      {abbr}
    </span>
  );
}

function getPreset(id) { return presets.find((p) => p.id === id) ?? presets[1]; }

function confidence(predictions) {
  if (!predictions || predictions.length < 2) return null;
  const top = predictions[0].champion_probability;
  const gap = top - predictions[1].champion_probability;
  if (top > 0.6 || gap > 0.4) return { label: "High Confidence", color: "#00a651", bg: "#f0fff6", uncertain: false };
  if (top > 0.35 || gap > 0.2) return { label: "Moderate",        color: "#f5a623", bg: "#fffbf0", uncertain: false };
  return                               { label: "Uncertain",        color: "#e10600", bg: "#fff5f5", uncertain: true  };
}

// Blend model probabilities with normalised historical scores
function computeAdjusted(predictions, stats) {
  const ALPHA = 0.38; // historical weight when uncertain
  const statsMap = Object.fromEntries(stats.map((s) => [s.constructor_name, s]));

  const enriched = predictions.map((p) => {
    const s = statsMap[p.constructor_name] ?? { championships: 0, seasons: 1, avg_position_5yr: 10, points_trend: 0 };
    return { ...p, _s: s };
  });

  // Normalise championship rate
  const champRates = enriched.map((p) => p._s.championships / Math.max(p._s.seasons, 1));
  const maxCR = Math.max(...champRates, 0.001);
  const normCR = champRates.map((r) => r / maxCR);

  // Normalise recent form (invert avg position — lower is better)
  const forms = enriched.map((p) => 1 / Math.max(p._s.avg_position_5yr, 1));
  const maxF  = Math.max(...forms, 0.001);
  const normF = forms.map((f) => f / maxF);

  // Normalise points trend
  const trends = enriched.map((p) => p._s.points_trend);
  const minT = Math.min(...trends), maxT = Math.max(...trends);
  const normT = trends.map((t) => maxT === minT ? 0.5 : (t - minT) / (maxT - minT));

  // Composite historical score
  const histScores = enriched.map((_, i) => 0.50 * normCR[i] + 0.35 * normF[i] + 0.15 * normT[i]);
  const maxH = Math.max(...histScores, 0.001);
  const normH = histScores.map((h) => h / maxH);

  // Multiplicative blend: history acts as a modifier on the model probability,
  // never as an additive override. A team with near-zero model probability
  // (e.g. 0 wins entered manually) stays near-zero regardless of history.
  const raw = enriched.map((p, i) => ({
    constructor_name:  p.constructor_name,
    model_probability: p.champion_probability,
    historical_score:  normH[i],
    _adjusted:         p.champion_probability * (1 - ALPHA * (1 - normH[i])),
    championships:     p._s.championships,
    avg_position_5yr:  p._s.avg_position_5yr,
  }));

  const totalAdj = raw.reduce((s, r) => s + r._adjusted, 0) || 1;

  return raw
    .map((r) => ({ ...r, champion_probability: r._adjusted / totalAdj }))
    .sort((a, b) => b.champion_probability - a.champion_probability);
}

// ── Small components ──────────────────────────────────────────────────────────
function Skeleton() {
  return (
    <div className="skeleton-rows">
      {[...Array(8)].map((_, i) => (
        <div key={i} className="skeleton-row">
          <div className="skeleton-cell" style={{ width: "42%" }} />
          <div className="skeleton-cell" style={{ width: "16%" }} />
          <div className="skeleton-cell" style={{ width: "16%" }} />
        </div>
      ))}
    </div>
  );
}

function RunSummary({ title, presetId, result }) {
  const preset = getPreset(presetId);
  return (
    <article className="compare-card">
      <div className="compare-head">
        <div>
          <p className="eyebrow small">{title}</p>
          <h3>{preset.name} Model</h3>
          <p className="panel-note compact">{preset.note}</p>
        </div>
        {result && (
          <div className="compare-metrics">
            <span>{result.accuracy.toFixed(3)} accuracy</span>
            <span>{result.top1_accuracy === null ? "n/a top-1" : `${result.top1_accuracy.toFixed(3)} top-1`}</span>
          </div>
        )}
      </div>
      {!result && <p className="panel-note">Run this model to see results.</p>}
    </article>
  );
}

// ── Upset Detector ────────────────────────────────────────────────────────────
function UpsetDetector({ allPredictions, picks }) {
  if (!picks?.length) return null;

  const missed = picks.filter((p) => p.is_true_champion === 0);
  if (!missed.length) {
    return (
      <div className="chart-block">
        <h3>Upset Detector</h3>
        <p className="muted">No upsets — the model correctly called every test season.</p>
      </div>
    );
  }

  const upsets = missed
    .map((p) => {
      const yr = Number(p.year);
      const yearPreds = allPredictions
        .filter((r) => Number(r.year) === yr)
        .sort((a, b) => Number(b.y_proba) - Number(a.y_proba));
      const actualChamp = yearPreds.find((r) => r.y_true === 1);
      const modelPickProb  = Number(p.predicted_prob);
      const actualChampProb = actualChamp ? Number(actualChamp.y_proba) : 0;
      return {
        year: yr,
        modelPick: p.constructor_name,
        modelPickProb,
        actualChamp: actualChamp?.constructor_name ?? "Unknown",
        actualChampProb,
        upsetScore: modelPickProb - actualChampProb,
      };
    })
    .sort((a, b) => b.upsetScore - a.upsetScore);

  return (
    <div className="chart-block">
      <h3>Upset Detector</h3>
      <p className="panel-note compact">
        Seasons where the model predicted the wrong champion, ranked by how confident it was in the
        incorrect pick. A large confidence gap means the model was very sure — and very wrong.
      </p>
      <div className="upset-grid">
        {upsets.map((u) => {
          const pickColor  = teamColors[u.modelPick]  || "#9AA0A6";
          const champColor = teamColors[u.actualChamp] || "#9AA0A6";
          const gapPct = Math.min(u.upsetScore * 100, 100);
          return (
            <article key={u.year} className="upset-card">
              <div className="upset-year-row">
                <span className="upset-year">{u.year}</span>
                <span className="upset-gap-label">{gapPct.toFixed(1)}pp gap</span>
              </div>
              <div className="upset-teams">
                <div className="upset-team">
                  <span className="eyebrow small" style={{ color: "#e10600" }}>Model picked</span>
                  <div className="upset-team-name">
                    <TeamLogo name={u.modelPick} color={pickColor} size={20} />
                    <span>{shortName(u.modelPick)}</span>
                  </div>
                  <span className="upset-prob" style={{ color: "#e10600" }}>
                    {(u.modelPickProb * 100).toFixed(1)}%
                  </span>
                </div>
                <span className="upset-arrow">→</span>
                <div className="upset-team">
                  <span className="eyebrow small" style={{ color: "#00a651" }}>Actual champion</span>
                  <div className="upset-team-name">
                    <TeamLogo name={u.actualChamp} color={champColor} size={20} />
                    <span>{shortName(u.actualChamp)}</span>
                  </div>
                  <span className="upset-prob" style={{ color: "#00a651" }}>
                    {(u.actualChampProb * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
              <div className="upset-bar-row">
                <div className="upset-bar-track">
                  <div className="upset-bar-fill" style={{ width: `${gapPct}%` }} />
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

// ── Overview: team probability line chart ─────────────────────────────────────
function TeamProbChart({ allPredictions }) {
  if (!allPredictions?.length) return null;

  const byYear = {};
  allPredictions.forEach((row) => {
    const yr = Number(row.year);
    if (!byYear[yr]) byYear[yr] = [];
    byYear[yr].push(row);
  });
  const years = Object.keys(byYear).map(Number).sort((a, b) => a - b);
  if (years.length < 2) return null;

  // Top 8 teams by peak probability in the test period
  const teamPeak = {};
  allPredictions.forEach((row) => {
    const p = Number(row.y_proba);
    if (!teamPeak[row.constructor_name] || p > teamPeak[row.constructor_name]) teamPeak[row.constructor_name] = p;
  });
  const topTeams = Object.entries(teamPeak).sort((a, b) => b[1] - a[1]).slice(0, 8).map(([n]) => n);

  const W = 660, H = 280;
  const pad = { top: 20, right: 150, bottom: 40, left: 50 };
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;
  const xPos = (yr) => years.length < 2 ? 0 : ((yr - years[0]) / (years[years.length - 1] - years[0])) * cw;
  const yPos = (p) => ch - Number(p) * ch;
  const gridLines = [0, 0.25, 0.5, 0.75, 1];
  const yearStep = Math.ceil(years.length / 8);

  return (
    <div className="chart-block">
      <h3>Championship Probability by Team — Test Seasons</h3>
      <p className="panel-note compact">
        Each line traces a constructor's predicted championship probability across every held-out test
        season. Teams near the top of the chart were rated as strong title contenders by the model.
        Filled circles mark the <strong>actual champion</strong> for that year.
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} className="forecast-svg" style={{ marginTop: 16 }}>
        <g transform={`translate(${pad.left},${pad.top})`}>
          {gridLines.map((v) => (
            <g key={v}>
              <line x1={0} y1={yPos(v)} x2={cw} y2={yPos(v)} stroke="#e1d7c8" strokeDasharray="4 3" />
              <text x={-8} y={yPos(v) + 4} textAnchor="end" fontSize={10} fill="#8a8d94">{Math.round(v * 100)}%</text>
            </g>
          ))}
          {years.filter((_, i) => i % yearStep === 0).map((yr) => (
            <text key={yr} x={xPos(yr)} y={ch + 22} textAnchor="middle" fontSize={10} fill="#5c5f66">{yr}</text>
          ))}
          {topTeams.map((team) => {
            const color = teamColors[team] || "#9AA0A6";
            const pts = years
              .map((yr) => {
                const row = byYear[yr]?.find((r) => r.constructor_name === team);
                return row ? `${xPos(yr).toFixed(1)},${yPos(row.y_proba).toFixed(1)}` : null;
              })
              .filter(Boolean);
            if (pts.length < 2) return null;
            const lastYr = years.slice().reverse().find((yr) => byYear[yr]?.find((r) => r.constructor_name === team));
            const lastRow = lastYr != null ? byYear[lastYr].find((r) => r.constructor_name === team) : null;
            return (
              <g key={team}>
                <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" strokeOpacity={0.85} />
                {lastRow && (
                  <text x={cw + 6} y={yPos(lastRow.y_proba) + 4} fontSize={11} fill={color}>{shortName(team)}</text>
                )}
              </g>
            );
          })}
          {/* Actual champion dots */}
          {allPredictions
            .filter((r) => r.y_true === 1 && topTeams.includes(r.constructor_name))
            .map((r) => {
              const color = teamColors[r.constructor_name] || "#9AA0A6";
              return (
                <circle
                  key={`${r.year}-${r.constructor_name}`}
                  cx={xPos(Number(r.year))} cy={yPos(r.y_proba)}
                  r={5} fill={color} stroke="white" strokeWidth={1.5}
                />
              );
            })}
        </g>
      </svg>
      <div className="chart-legend" style={{ marginTop: 8 }}>
        <span>
          <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: "50%", background: "#555", border: "1.5px solid white", marginRight: 6, verticalAlign: "middle" }} />
          Filled circle = actual champion that year
        </span>
      </div>
    </div>
  );
}

// ── Overview: season-by-season breakdown cards ────────────────────────────────
function SeasonBreakdown({ picks, allPredictions }) {
  if (!picks?.length) return null;

  const byYear = {};
  allPredictions.forEach((row) => {
    const yr = Number(row.year);
    if (!byYear[yr]) byYear[yr] = [];
    byYear[yr].push(row);
  });
  Object.values(byYear).forEach((rows) => rows.sort((a, b) => Number(b.y_proba) - Number(a.y_proba)));

  return (
    <div className="chart-block">
      <h3>Season-by-Season Breakdown</h3>
      <p className="panel-note compact">
        Each card covers one test season. Bars show every constructor's predicted championship
        probability (scaled to the season leader). The <strong style={{ color: "var(--accent-gold)" }}>★</strong> marks
        the actual champion. A <span style={{ color: "#00a651", fontWeight: 600 }}>green</span> header means
        the model's top pick was correct; <span style={{ color: "#e10600", fontWeight: 600 }}>red</span> means it chose the wrong team.
      </p>
      <div className="season-grid">
        {picks.map((pick) => {
          const yr = Number(pick.year);
          const teams = byYear[yr] || [];
          const correct = pick.is_true_champion === 1;
          const maxProb = Math.max(...teams.map((t) => Number(t.y_proba)), 0.001);
          return (
            <article key={yr} className={`season-card ${correct ? "season-card--correct" : "season-card--wrong"}`}>
              <div className="season-card-head">
                <span className="season-year">{yr}</span>
                <span className={correct ? "verdict-correct" : "verdict-wrong"}>{correct ? "✓ Correct" : "✗ Missed"}</span>
              </div>
              <div className="season-pick">
                <span className="muted" style={{ fontSize: 11 }}>Top pick:</span>
                <strong style={{ color: teamColors[pick.constructor_name] || "var(--accent)", fontSize: 13 }}>
                  {shortName(pick.constructor_name)}
                </strong>
                <span className="muted" style={{ fontSize: 11 }}>{(Number(pick.predicted_prob) * 100).toFixed(1)}%</span>
              </div>
              <div className="season-bars">
                {teams.slice(0, 6).map((t) => {
                  const isChamp = t.y_true === 1;
                  const color = teamColors[t.constructor_name] || "#9AA0A6";
                  const pct = (Number(t.y_proba) / maxProb) * 100;
                  return (
                    <div key={t.constructor_name} className="season-bar-row">
                      <span className="season-team-name">
                        {isChamp && <span className="champion-star">★</span>}
                        {shortName(t.constructor_name)}
                      </span>
                      <div className="season-bar-track">
                        <div className="season-bar-fill" style={{ width: `${pct}%`, background: color, opacity: isChamp ? 1 : 0.5 }} />
                      </div>
                      <span className="season-prob">{(Number(t.y_proba) * 100).toFixed(0)}%</span>
                    </div>
                  );
                })}
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

// ── Forecast SVG line chart ───────────────────────────────────────────────────
function ForecastChart({ forecastYears }) {
  const years = forecastYears.map((fy) => fy.year);
  if (years.length < 2) return null;

  // Gather top teams by avg probability
  const teamProbs = {};
  forecastYears.forEach((fy) => {
    fy.rows.forEach((row) => {
      if (!teamProbs[row.constructor_name]) teamProbs[row.constructor_name] = [];
      teamProbs[row.constructor_name].push(Number(row.champion_probability));
    });
  });
  const topTeams = Object.entries(teamProbs)
    .map(([name, probs]) => ({ name, avg: probs.reduce((s, p) => s + p, 0) / probs.length }))
    .sort((a, b) => b.avg - a.avg)
    .slice(0, 7)
    .map((t) => t.name);

  const W = 620, H = 260;
  const pad = { top: 16, right: 140, bottom: 36, left: 44 };
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;

  const xPos = (year) => (years.indexOf(year) / (years.length - 1)) * cw;
  const yPos = (prob) => ch - Number(prob) * ch;

  const gridLines = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="forecast-chart-wrap">
      <h3>Projected Championship Probability</h3>
      <svg viewBox={`0 0 ${W} ${H}`} className="forecast-svg">
        <g transform={`translate(${pad.left},${pad.top})`}>
          {gridLines.map((v) => (
            <g key={v}>
              <line x1={0} y1={yPos(v)} x2={cw} y2={yPos(v)} stroke="#e1d7c8" strokeDasharray="4 3" />
              <text x={-8} y={yPos(v) + 4} textAnchor="end" fontSize={10} fill="#8a8d94">{Math.round(v * 100)}%</text>
            </g>
          ))}
          {years.map((yr) => (
            <text key={yr} x={xPos(yr)} y={ch + 20} textAnchor="middle" fontSize={11} fill="#5c5f66">{yr}</text>
          ))}
          {topTeams.map((team) => {
            const color = teamColors[team] || "#9AA0A6";
            const pts = forecastYears
              .map((fy) => fy.rows.find((r) => r.constructor_name === team))
              .filter(Boolean)
              .map((r, i) => `${xPos(years[i])},${yPos(r.champion_probability)}`);
            if (pts.length < 2) return null;
            const lastFy = forecastYears[forecastYears.length - 1];
            const lastRow = lastFy.rows.find((r) => r.constructor_name === team);
            return (
              <g key={team}>
                <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth={2.5} strokeLinejoin="round" />
                {lastRow && (
                  <text x={cw + 6} y={yPos(lastRow.champion_probability) + 4} fontSize={11} fill={color}>{shortName(team)}</text>
                )}
              </g>
            );
          })}
        </g>
      </svg>
    </div>
  );
}

// ── Race Breakdown: SVG line chart ────────────────────────────────────────────
const confColors = { High: "#00a651", Moderate: "#f5a623", Uncertain: "#e10600" };
const confBgs    = { High: "#f0fff6", Moderate: "#fffbf0", Uncertain: "#fff5f5" };

function RaceBreakdownChart({ rounds }) {
  if (!rounds?.length) return null;

  const roundNums = rounds.map((r) => r.round);

  const teamPeak = {};
  rounds.forEach((r) => {
    r.predictions.forEach((p) => {
      if (!teamPeak[p.constructor_name] || p.champion_probability > teamPeak[p.constructor_name])
        teamPeak[p.constructor_name] = p.champion_probability;
    });
  });
  const topTeams = Object.entries(teamPeak)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([n]) => n);

  const W = 680, H = 310;
  const confH = 16;
  const pad = { top: 16, right: 150, bottom: 56 + confH, left: 44 };
  const cw = W - pad.left - pad.right;
  const ch = H - pad.top - pad.bottom;

  const minRound = Math.min(...roundNums);
  const maxRound = Math.max(...roundNums);
  const xPos = (rnd) =>
    roundNums.length < 2 ? 0 : ((rnd - minRound) / (maxRound - minRound)) * cw;
  const yPos = (prob) => ch - Number(prob) * ch;

  const roundStep = Math.ceil(roundNums.length / 10);
  const gridLines = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="chart-block">
      <h3>Championship Probability — Round by Round</h3>
      <p className="panel-note compact">
        Each line shows how a constructor's predicted championship probability evolves race by race.
        The colour strip below the X-axis indicates model confidence at each round.
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} className="forecast-svg" style={{ marginTop: 16 }}>
        <g transform={`translate(${pad.left},${pad.top})`}>
          {gridLines.map((v) => (
            <g key={v}>
              <line x1={0} y1={yPos(v)} x2={cw} y2={yPos(v)} stroke="#e1d7c8" strokeDasharray="4 3" />
              <text x={-8} y={yPos(v) + 4} textAnchor="end" fontSize={10} fill="#8a8d94">{Math.round(v * 100)}%</text>
            </g>
          ))}

          {roundNums.filter((_, i) => i % roundStep === 0).map((rnd) => (
            <text key={rnd} x={xPos(rnd)} y={ch + 20} textAnchor="middle" fontSize={10} fill="#5c5f66">R{rnd}</text>
          ))}

          {/* Confidence strip */}
          {rounds.map((r, i) => {
            const x0 = xPos(r.round);
            const x1 = i < rounds.length - 1 ? xPos(rounds[i + 1].round) : cw;
            return (
              <rect
                key={r.round}
                x={x0} y={ch + 30}
                width={Math.max(x1 - x0 - 1, 1)} height={confH}
                fill={confColors[r.confidence] || "#9AA0A6"}
                opacity={0.55} rx={2}
              />
            );
          })}

          {topTeams.map((team) => {
            const color = teamColors[team] || "#9AA0A6";
            const pts = rounds
              .map((r) => {
                const p = r.predictions.find((p) => p.constructor_name === team);
                return p ? `${xPos(r.round).toFixed(1)},${yPos(p.champion_probability).toFixed(1)}` : null;
              })
              .filter(Boolean);
            if (pts.length < 2) return null;
            const lastPred = rounds[rounds.length - 1]?.predictions.find((p) => p.constructor_name === team);
            return (
              <g key={team}>
                <polyline points={pts.join(" ")} fill="none" stroke={color} strokeWidth={2.2} strokeLinejoin="round" strokeOpacity={0.9} />
                {lastPred && (
                  <text x={cw + 6} y={yPos(lastPred.champion_probability) + 4} fontSize={11} fill={color}>
                    {shortName(team)}
                  </text>
                )}
              </g>
            );
          })}
        </g>
      </svg>

      <div className="chart-legend" style={{ marginTop: 4, gap: 16 }}>
        {Object.entries(confColors).map(([label, color]) => (
          <span key={label}>
            <span style={{ display: "inline-block", width: 14, height: 8, background: color, opacity: 0.55, borderRadius: 2, marginRight: 4, verticalAlign: "middle" }} />
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}

// ── Race Breakdown: round-by-round table ──────────────────────────────────────
function RaceBreakdownTable({ rounds }) {
  if (!rounds?.length) return null;

  return (
    <div className="chart-block">
      <h3>Round-by-Round Summary</h3>
      <div className="breakdown-table-wrap">
        <table className="breakdown-table">
          <thead>
            <tr>
              <th>Round</th>
              <th>Model Pick</th>
              <th>Probability</th>
              <th>2nd Place</th>
              <th>Margin</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {rounds.map((r) => {
              const top    = r.predictions[0];
              const second = r.predictions[1];
              const margin = top && second ? top.champion_probability - second.champion_probability : 0;
              const topColor = top    ? (teamColors[top.constructor_name]    || "#9AA0A6") : "#9AA0A6";
              const secColor = second ? (teamColors[second.constructor_name] || "#9AA0A6") : "#9AA0A6";
              return (
                <tr key={r.round}>
                  <td className="breakdown-round">R{r.round}</td>
                  <td>
                    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                      {top && <TeamLogo name={top.constructor_name} color={topColor} size={18} />}
                      <span style={{ color: topColor, fontWeight: 600 }}>{shortName(top?.constructor_name || "")}</span>
                    </div>
                  </td>
                  <td className="breakdown-prob">{(r.top_probability * 100).toFixed(1)}%</td>
                  <td>
                    {second && (
                      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                        <TeamLogo name={second.constructor_name} color={secColor} size={16} />
                        <span style={{ color: secColor, fontSize: 12 }}>{shortName(second.constructor_name)}</span>
                      </div>
                    )}
                  </td>
                  <td style={{ fontSize: 12, color: margin > 0.2 ? "#00a651" : margin > 0.1 ? "#f5a623" : "var(--muted)", fontWeight: 600 }}>
                    {(margin * 100).toFixed(1)}pp
                  </td>
                  <td>
                    <span className="confidence-badge" style={{ background: confBgs[r.confidence] || "#f5f5f5", color: confColors[r.confidence] || "#5c5f66", fontSize: 11 }}>
                      {r.confidence}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [page, setPage]                   = useState("predict");
  const [darkMode, setDarkMode]           = useState(false);
  const [selectedPreset, setSelectedPreset] = useState("balanced");

  // Primary model run
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState("");
  const [result, setResult]   = useState(null);

  // Saved predictions
  const [savedPredictions, setSavedPredictions] = useState(() => {
    try { return JSON.parse(localStorage.getItem("f1_saved_preds") || "[]"); } catch { return []; }
  });
  const [saveName, setSaveName]               = useState("");
  const [showSaveInput, setShowSaveInput]     = useState(false);
  const [compareIds, setCompareIds]           = useState([]);

  // Forecast
  const [forecastForm, setForecastForm]       = useState({ years_ahead: 3, carry_forward: 0.7 });
  const [forecastResult, setForecastResult]   = useState(null);
  const [forecastLoading, setForecastLoading] = useState(false);
  const [forecastError, setForecastError]     = useState("");

  // Predict
  const [predictRows, setPredictRows]             = useState(null);
  const [originalRows, setOriginalRows]           = useState(null);
  const [predictSeason, setPredictSeason]         = useState(null);
  const [predictResult, setPredictResult]         = useState(null);
  const [prevPredictResult, setPrevPredictResult] = useState(null);
  const [predictLoading, setPredictLoading]       = useState(false);
  const [predictError, setPredictError]           = useState("");
  const [adjustedResult, setAdjustedResult]       = useState(null);
  const [adjustedLoading, setAdjustedLoading]     = useState(false);

  // Race breakdown
  const [breakdownSeason, setBreakdownSeason] = useState(null);
  const [breakdownResult, setBreakdownResult] = useState(null);
  const [breakdownLoading, setBreakdownLoading] = useState(false);
  const [breakdownError, setBreakdownError] = useState("");

  // Data source
  const [dataMode, setDataMode]           = useState("rounds");
  const [seasons, setSeasons]             = useState([]);
  const [midSeason, setMidSeason]         = useState({ season: null, round: 24, maxRound: 24 });
  const [midFetching, setMidFetching]     = useState(false);
  const [liveFetching, setLiveFetching]   = useState(false);
  const [liveError, setLiveError]         = useState("");

  useEffect(() => {
    fetchSeasons();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    document.body.setAttribute("data-theme", darkMode ? "dark" : "light");
  }, [darkMode]);

  // ── Data fetching ────────────────────────────────────────────────────────────
  const fetchSeasons = async () => {
    try {
      const res  = await fetch(`${API_BASE}/seasons`);
      const data = await res.json();
      setSeasons(data.seasons);
      if (data.seasons.length) {
        const latest = data.seasons[0];
        setMidSeason((p) => ({ ...p, season: latest.season, maxRound: latest.max_round }));
        setBreakdownSeason(latest.season);
        await loadMidSeasonStandings(latest.season, 24);
      }
    } catch (_) {}
  };

  const loadLiveStandings = async () => {
    setLiveFetching(true); setLiveError(""); setPredictError("");
    try {
      const res  = await fetch(`${API_BASE}/live-standings`);
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      if (!data.rows?.length) throw new Error("No live standings available yet");
      setPredictRows(data.rows);
      setOriginalRows(data.rows);
      setPredictSeason(`${data.season} — Round ${data.round} (Live)`);
      setPredictResult(null);
      setPrevPredictResult(null);
    } catch (err) { setLiveError(err.message); }
    finally { setLiveFetching(false); }
  };

  const loadMidSeasonStandings = async (seasonArg, roundArg) => {
    const season = seasonArg ?? midSeason.season;
    const round  = roundArg  ?? midSeason.round;
    if (!season) return;
    setDataMode("rounds");
    setMidFetching(true); setPredictError("");
    try {
      const res  = await fetch(`${API_BASE}/standings/${season}/${round}`);
      if (!res.ok) throw new Error("Failed to load standings");
      const data = await res.json();
      setPredictRows(data.rows);
      setOriginalRows(data.rows);
      setPredictSeason(`${data.season} — Round ${data.round}`);
      setPredictResult(null);
      setPrevPredictResult(null);
    } catch (err) { setPredictError(err.message); }
    finally { setMidFetching(false); }
  };

  // ── Handlers ─────────────────────────────────────────────────────────────────
  const runTraining = async () => {
    setLoading(true); setError("");
    try {
      const res  = await fetch(`${API_BASE}/train`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(getPreset(selectedPreset).config),
      });
      if (!res.ok) throw new Error(await res.text() || "Training failed");
      setResult(await res.json());
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  const runForecast = async () => {
    setForecastLoading(true); setForecastError("");
    try {
      const res = await fetch(`${API_BASE}/forecast`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...getPreset(selectedPreset).config,
          years_ahead:    Number(forecastForm.years_ahead),
          carry_forward:  Number(forecastForm.carry_forward),
        }),
      });
      if (!res.ok) throw new Error(await res.text() || "Forecast failed");
      setForecastResult(await res.json());
      setPage("forecast");
    } catch (err) { setForecastError(err.message); }
    finally { setForecastLoading(false); }
  };

  const updatePredictRow = (index, field, value) =>
    setPredictRows((prev) => prev.map((row, i) => i === index ? { ...row, [field]: value } : row));

  const randomizeStats = () => {
    const totalRounds = dataMode === "rounds" ? midSeason.round : 24;
    setPredictRows((prev) => {
      const wins = Array(prev.length).fill(0);
      for (let i = 0; i < totalRounds; i++) {
        wins[Math.floor(Math.random() * prev.length)]++;
      }
      return prev.map((row, i) => ({ ...row, wins_prev: wins[i] }));
    });
    setPredictResult(null);
  };

  const resetStats = () => {
    setPredictRows(originalRows);
    setPredictResult(null);
    setPrevPredictResult(null);
  };

  const saveSnapshot = () => {
    const snap = {
      id: Date.now(),
      label: saveName.trim() || (predictSeason ?? "Prediction"),
      season: predictSeason,
      preset: selectedPreset,
      timestamp: new Date().toISOString(),
      result: predictResult,
    };
    const updated = [snap, ...savedPredictions].slice(0, 20);
    setSavedPredictions(updated);
    localStorage.setItem("f1_saved_preds", JSON.stringify(updated));
    setSaveName("");
    setShowSaveInput(false);
  };

  const deleteSnapshot = (id) => {
    const updated = savedPredictions.filter((s) => s.id !== id);
    setSavedPredictions(updated);
    localStorage.setItem("f1_saved_preds", JSON.stringify(updated));
    setCompareIds((prev) => prev.filter((i) => i !== id));
  };

  const toggleCompare = (id) => {
    setCompareIds((prev) => {
      if (prev.includes(id)) return prev.filter((i) => i !== id);
      if (prev.length >= 2)  return [prev[1], id];
      return [...prev, id];
    });
  };

  const runBreakdown = async () => {
    if (!breakdownSeason) return;
    setBreakdownLoading(true); setBreakdownError(""); setBreakdownResult(null);
    try {
      const { n_estimators, max_depth, random_state, class_weight } = getPreset(selectedPreset).config;
      const res = await fetch(`${API_BASE}/race-breakdown`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ season: breakdownSeason, n_estimators, max_depth, random_state, class_weight }),
      });
      if (!res.ok) throw new Error("Race breakdown failed");
      setBreakdownResult(await res.json());
    } catch (err) { setBreakdownError(err.message); }
    finally { setBreakdownLoading(false); }
  };

  const runPredict = async () => {
    setPredictLoading(true); setPredictError("");
    setPrevPredictResult(predictResult);
    try {
      const { n_estimators, max_depth, random_state, class_weight } = getPreset(selectedPreset).config;
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows: predictRows, n_estimators, max_depth, random_state, class_weight }),
      });
      if (!res.ok) throw new Error("Prediction failed");
      const data = await res.json();
      setPredictResult(data.predictions);
    } catch (err) { setPredictError(err.message); }
    finally { setPredictLoading(false); }
  };

  // Auto-fetch historical stats and compute adjusted result when model is uncertain
  useEffect(() => {
    if (!predictResult) { setAdjustedResult(null); return; }
    const conf = confidence(predictResult);
    if (!conf?.uncertain) { setAdjustedResult(null); return; }

    setAdjustedLoading(true);
    const names = predictResult.map((r) => r.constructor_name).join(",");
    fetch(`${API_BASE}/constructor-stats?names=${encodeURIComponent(names)}`)
      .then((res) => res.json())
      .then((data) => setAdjustedResult(computeAdjusted(predictResult, data.stats)))
      .catch(() => setAdjustedResult(null))
      .finally(() => setAdjustedLoading(false));
  }, [predictResult]);

  // ── Derived ──────────────────────────────────────────────────────────────────
  const conf = useMemo(() => confidence(predictResult), [predictResult]);

  const deltaMap = useMemo(() => {
    if (!prevPredictResult || !predictResult) return {};
    const map = {};
    prevPredictResult.forEach((r) => { map[r.constructor_name] = r.champion_probability; });
    return map;
  }, [prevPredictResult, predictResult]);

  // ── Render ───────────────────────────────────────────────────────────────────
  return (
    <div className="page">

      {/* Hero */}
      <header className="hero">
        <div>
          <p className="eyebrow">F1 ML Lab</p>
          <h1>Constructor Champion Predictor</h1>
          <p className="subtitle">
            Explore constructor standings, forecast future seasons, and predict who takes the championship.
          </p>
        </div>
        <div className="hero-card">
          <p>API Status</p>
          <a href={`${API_BASE}/health`} target="_blank" rel="noreferrer">localhost:8000</a>
        </div>
      </header>

      {/* Nav */}
      <nav className="nav-shell" aria-label="Main navigation">
        {pages.map((item) => (
          <button key={item.id} type="button"
            className={page === item.id ? "nav-pill active" : "nav-pill"}
            onClick={() => setPage(item.id)}
          >
            {item.label}
          </button>
        ))}
        <button
          type="button"
          className="nav-pill dark-toggle"
          onClick={() => setDarkMode((d) => !d)}
          title={darkMode ? "Switch to light mode" : "Switch to dark mode"}
          style={{ marginLeft: "auto" }}
        >
          {darkMode ? "☀ Light" : "☾ Dark"}
        </button>
      </nav>

      {/* ══ PREDICT ══════════════════════════════════════════════════════════ */}
      {page === "predict" && (
        <section className="panel panel-highlight">

          {/* Header */}
          <div className="panel-header">
            <div>
              <h2>Predict Next Champion</h2>
            </div>
            <div className="action-row">
              <button className="primary secondary" onClick={resetStats} disabled={!originalRows}>Reset</button>
              <button className="primary secondary" onClick={randomizeStats} disabled={!predictRows}>Randomize</button>
              <button className="primary" onClick={runPredict} disabled={predictLoading || !predictRows}>
                {predictLoading ? "Predicting…" : "Predict Champion"}
              </button>
              {predictResult && !showSaveInput && (
                <button className="primary secondary" onClick={() => setShowSaveInput(true)}>Save Snapshot</button>
              )}
              {predictResult && showSaveInput && (
                <div className="save-input-row">
                  <input
                    type="text"
                    placeholder={predictSeason ?? "Snapshot name"}
                    value={saveName}
                    onChange={(e) => setSaveName(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && saveSnapshot()}
                    autoFocus
                  />
                  <button className="primary" onClick={saveSnapshot}>Save</button>
                  <button className="primary secondary" onClick={() => { setShowSaveInput(false); setSaveName(""); }}>Cancel</button>
                </div>
              )}
            </div>
          </div>

          {/* Model + data source bar */}
          <div className="data-source-bar">
            <span className="data-source-label">Model:</span>
            {presets.map((p) => (
              <button key={p.id} type="button" title={p.note}
                className={selectedPreset === p.id ? "source-pill active" : "source-pill"}
                onClick={() => setSelectedPreset(p.id)}
              >
                {p.name}
              </button>
            ))}
            <div className="nav-divider" />
            <span className="data-source-label">Data source:</span>
            <div className="midseason-controls">
              <label>
                Rounds completed
                <input type="number" min={1}
                  value={midSeason.round}
                  onChange={(e) => setMidSeason((p) => ({ ...p, round: Math.max(1, Number(e.target.value)) }))}
                />
              </label>
              <button className="primary secondary load-btn" onClick={() => loadMidSeasonStandings()} disabled={midFetching}>
                {midFetching ? "Loading…" : "Load"}
              </button>
            </div>
            <button
              className={dataMode === "live" ? "source-pill live-pill active" : "source-pill live-pill"}
              onClick={() => { setDataMode("live"); loadLiveStandings(); }}
              disabled={liveFetching}
            >
              <span className="live-dot" />
              {liveFetching ? "Loading…" : "Live 2026 Season"}
            </button>
          </div>

          {predictError && <p className="error">{predictError}</p>}
          {liveError && <p className="error">Live data: {liveError}</p>}

          {/* Main layout */}
          <div className="predict-layout">

            {/* Left: editable table */}
            <div>
              {midFetching ? <Skeleton /> : predictRows && (
                <table className="predict-table">
                  <thead><tr><th>Constructor</th><th>Wins</th></tr></thead>
                  <tbody>
                    {predictRows.map((row, i) => (
                      <tr key={row.constructor_name}>
                        <td>
                          <TeamLogo name={row.constructor_name} color={teamColors[row.constructor_name] || "#9AA0A6"} size={20} />
                          {" "}{row.constructor_name}
                        </td>
                        <td>
                          <input type="number" min="0" max="24" value={row.wins_prev}
                            onChange={(e) => updatePredictRow(i, "wins_prev", Number(e.target.value))} />
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>

            {/* Right: results */}
            {predictResult ? (
              <div className="predict-results">
                {/* Winner card */}
                <div className="champion-callout predict-winner" key={predictResult[0].constructor_name}>
                  <span>Predicted Champion</span>
                  <div className="champion-name-row">
                    <TeamLogo name={predictResult[0].constructor_name} color={teamColors[predictResult[0].constructor_name] || "#9AA0A6"} size={34} />
                    <strong>{predictResult[0].constructor_name}</strong>
                  </div>
                  <span className="prob-label">
                    {(predictResult[0].champion_probability * 100).toFixed(1)}% probability
                  </span>
                  {conf && (
                    <span className="confidence-badge" style={{ background: conf.bg, color: conf.color }}>
                      {conf.label}
                    </span>
                  )}
                </div>

                {/* Bars */}
                <div className="bar-list">
                  {predictResult.map((row) => {
                    const color = teamColors[row.constructor_name] || "#9AA0A6";
                    const prev  = deltaMap[row.constructor_name];
                    const delta = prev !== undefined ? row.champion_probability - prev : null;
                    return (
                      <div key={row.constructor_name} className="bar-row">
                        <div className="bar-label">
                          <TeamLogo name={row.constructor_name} color={color} size={20} />
                          <span>{row.constructor_name}</span>
                        </div>
                        <div className="bar-track">
                          <div className="bar-fill animated" style={{ width: `${row.champion_probability * 100}%`, background: color }} />
                        </div>
                        <div className="bar-right">
                          <span className="bar-value">{(row.champion_probability * 100).toFixed(1)}%</span>
                          {delta !== null && Math.abs(delta) > 0.001 && (
                            <span className="delta" style={{ color: delta > 0 ? "#00a651" : "#e10600" }}>
                              {delta > 0 ? "▲" : "▼"}{(Math.abs(delta) * 100).toFixed(1)}%
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>

                {prevPredictResult && (
                  <p className="panel-note" style={{ marginTop: 8 }}>
                    Arrows show change from your previous prediction.
                  </p>
                )}

                {/* Data-adjusted panel (uncertain only) */}
                {conf?.uncertain && (
                  <div className="adjusted-panel">
                    <div className="adjusted-header">
                      <span className="eyebrow small">Data-Adjusted Prediction</span>
                      <p className="panel-note compact">
                        Model confidence is low — historical championships, recent form, and points
                        trend have been blended in to refine the ranking.
                      </p>
                    </div>
                    {adjustedLoading && <p className="muted" style={{ fontSize: 13 }}>Fetching historical data…</p>}
                    {adjustedResult && (
                      <>
                        <div className="adjusted-winner">
                          <span>Adjusted Pick</span>
                          <strong style={{ color: teamColors[adjustedResult[0].constructor_name] || "var(--accent)" }}>
                            {adjustedResult[0].constructor_name}
                          </strong>
                          {adjustedResult[0].constructor_name !== predictResult[0].constructor_name && (
                            <span className="adjusted-differs">differs from model pick</span>
                          )}
                        </div>
                        <div className="adjusted-rows">
                          {adjustedResult.slice(0, 6).map((row, rank) => {
                            const color = teamColors[row.constructor_name] || "#9AA0A6";
                            return (
                              <div key={row.constructor_name} className="adjusted-row">
                                <span className="adjusted-rank">{rank + 1}</span>
                                <TeamLogo name={row.constructor_name} color={color} size={18} />
                                <span className="adjusted-name">{row.constructor_name}</span>
                                <div className="adjusted-bars">
                                  <div className="adjusted-bar-wrap" title="Model probability">
                                    <div className="adjusted-bar model-bar" style={{ width: `${row.model_probability * 100}%`, background: color, opacity: 0.45 }} />
                                  </div>
                                  <div className="adjusted-bar-wrap" title="Adjusted probability">
                                    <div className="adjusted-bar" style={{ width: `${row.champion_probability * 100}%`, background: color }} />
                                  </div>
                                </div>
                                <div className="adjusted-meta">
                                  <span>{(row.champion_probability * 100).toFixed(1)}%</span>
                                  <span className="adjusted-sub">{row.championships} titles · avg pos {row.avg_position_5yr.toFixed(1)}</span>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                        <p className="adjusted-legend">Light bar = model only · Solid bar = model + history</p>
                      </>
                    )}
                  </div>
                )}
              </div>
            ) : (
              <div className="predict-placeholder">
                <p className="muted">Results will appear here after you click Predict Champion.</p>
              </div>
            )}
          </div>
        </section>
      )}

      {/* ══ RACE BREAKDOWN ═══════════════════════════════════════════════════ */}
      {page === "breakdown" && (
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Race-by-Race Breakdown</h2>
              <p className="panel-note compact">
                See how championship probability and confidence evolve round by round for any season.
              </p>
            </div>
            <div className="action-row">
              <label className="inline-label">
                Season
                <select
                  value={breakdownSeason || ""}
                  onChange={(e) => { setBreakdownSeason(Number(e.target.value)); setBreakdownResult(null); }}
                >
                  {seasons.map((s) => (
                    <option key={s.season} value={s.season}>{s.season}</option>
                  ))}
                </select>
              </label>
              {breakdownError && <p className="error">{breakdownError}</p>}
              <button className="primary" onClick={runBreakdown} disabled={breakdownLoading || !breakdownSeason}>
                {breakdownLoading ? "Analysing…" : "Run Breakdown"}
              </button>
            </div>
          </div>

          {!breakdownResult && !breakdownLoading && (
            <p className="muted" style={{ marginTop: 8 }}>
              Select a season and click <strong>Run Breakdown</strong> to see round-by-round championship probabilities.
            </p>
          )}
          {breakdownLoading && (
            <div className="skeleton-rows" style={{ marginTop: 16 }}>
              {[...Array(4)].map((_, i) => (
                <div key={i} className="skeleton-row">
                  <div className="skeleton-cell" style={{ width: "60%" }} />
                  <div className="skeleton-cell" style={{ width: "20%" }} />
                </div>
              ))}
            </div>
          )}
          {breakdownResult && (
            <>
              <div className="results-grid" style={{ marginTop: 20 }}>
                <div className="stat">
                  <p>Season</p>
                  <strong>{breakdownResult.season}</strong>
                </div>
                <div className="stat">
                  <p>Rounds</p>
                  <strong>{breakdownResult.rounds.length}</strong>
                  <span className="stat-note">rounds with standings data</span>
                </div>
                <div className="stat">
                  <p>Final Pick</p>
                  <strong style={{ color: teamColors[breakdownResult.rounds[breakdownResult.rounds.length - 1]?.top_pick] || "var(--accent)", fontSize: 16 }}>
                    {shortName(breakdownResult.rounds[breakdownResult.rounds.length - 1]?.top_pick || "")}
                  </strong>
                  <span className="stat-note">
                    {(breakdownResult.rounds[breakdownResult.rounds.length - 1]?.top_probability * 100).toFixed(1)}% final round probability
                  </span>
                </div>
                <div className="stat">
                  <p>Final Confidence</p>
                  <strong style={{ color: confColors[breakdownResult.rounds[breakdownResult.rounds.length - 1]?.confidence] || "var(--muted)" }}>
                    {breakdownResult.rounds[breakdownResult.rounds.length - 1]?.confidence}
                  </strong>
                </div>
              </div>
              <RaceBreakdownChart rounds={breakdownResult.rounds} />
              <RaceBreakdownTable rounds={breakdownResult.rounds} />
            </>
          )}
        </section>
      )}

      {/* ══ SAVED ════════════════════════════════════════════════════════════ */}
      {page === "saved" && (
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Saved Predictions</h2>
              <p className="panel-note compact">
                Save snapshots from the Predict page and select two to compare them side by side.
              </p>
            </div>
          </div>

          {savedPredictions.length === 0 && (
            <div className="overview-empty">
              <p className="muted">
                No snapshots yet. Go to <strong>Predict</strong>, run a prediction, and click <strong>Save Snapshot</strong>.
              </p>
            </div>
          )}

          {savedPredictions.length > 0 && (
            <>
              {compareIds.length === 2 && (() => {
                const snapA = savedPredictions.find((s) => s.id === compareIds[0]);
                const snapB = savedPredictions.find((s) => s.id === compareIds[1]);
                if (!snapA || !snapB) return null;

                const presetA  = getPreset(snapA.preset);
                const presetB  = getPreset(snapB.preset);
                const topProbA = snapA.result?.[0]?.champion_probability ?? 0;
                const topProbB = snapB.result?.[0]?.champion_probability ?? 0;
                const gapA     = topProbA - (snapA.result?.[1]?.champion_probability ?? 0);
                const gapB     = topProbB - (snapB.result?.[1]?.champion_probability ?? 0);
                const samePick = snapA.result?.[0]?.constructor_name === snapB.result?.[0]?.constructor_name;

                const allTeams = [...new Set([
                  ...snapA.result.map((r) => r.constructor_name),
                  ...snapB.result.map((r) => r.constructor_name),
                ])];
                const mapA = Object.fromEntries(snapA.result.map((r) => [r.constructor_name, r.champion_probability]));
                const mapB = Object.fromEntries(snapB.result.map((r) => [r.constructor_name, r.champion_probability]));
                const rows = allTeams
                  .map((name) => ({ name, probA: mapA[name] ?? 0, probB: mapB[name] ?? 0, delta: (mapB[name] ?? 0) - (mapA[name] ?? 0) }))
                  .sort((a, b) => b.probA - a.probA);

                return (
                  <div className="compare-view">
                    <div className="compare-view-header">
                      <div className="compare-view-label">
                        <span className="eyebrow small">Snapshot A</span>
                        <strong>{snapA.label}</strong>
                        <span className="muted" style={{ fontSize: 12 }}>{new Date(snapA.timestamp).toLocaleDateString()}</span>
                      </div>
                      <div className="compare-view-vs">vs</div>
                      <div className="compare-view-label compare-view-label--b">
                        <span className="eyebrow small">Snapshot B</span>
                        <strong>{snapB.label}</strong>
                        <span className="muted" style={{ fontSize: 12 }}>{new Date(snapB.timestamp).toLocaleDateString()}</span>
                      </div>
                    </div>
                    {/* Analysis blurb */}
                    <div className="compare-analysis">

                      {/* Model comparison */}
                      <div className="compare-analysis-block">
                        <p className="eyebrow small">Model</p>
                        {snapA.preset === snapB.preset ? (
                          <p>Both snapshots used the <strong>{presetA.name}</strong> model ({presetA.config.n_estimators} trees, depth {presetA.config.max_depth}). Differences reflect changes in the input standings only, not the model.</p>
                        ) : (
                          <p>
                            A used <strong>{presetA.name}</strong> ({presetA.config.n_estimators} trees, depth {presetA.config.max_depth}) and B used <strong>{presetB.name}</strong> ({presetB.config.n_estimators} trees, depth {presetB.config.max_depth}).{" "}
                            {presetB.config.n_estimators > presetA.config.n_estimators
                              ? "The deeper B model can capture more complex patterns but may be more sensitive to noise in the training data."
                              : "The shallower A model generalises more broadly; B may miss subtler competitive signals."}
                          </p>
                        )}
                      </div>

                      {/* Confidence comparison */}
                      <div className="compare-analysis-block">
                        <p className="eyebrow small">Confidence</p>
                        <div className="compare-confidence-row">
                          <div className="compare-confidence-item">
                            <span className="muted" style={{ fontSize: 12 }}>Snapshot A — top pick</span>
                            <strong style={{ fontSize: 22, color: teamColors[snapA.result?.[0]?.constructor_name] || "var(--accent)" }}>
                              {(topProbA * 100).toFixed(1)}%
                            </strong>
                            <span className="muted" style={{ fontSize: 12 }}>Margin over 2nd: <strong>{(gapA * 100).toFixed(1)}pp</strong></span>
                          </div>
                          <div className="compare-confidence-divider" />
                          <div className="compare-confidence-item">
                            <span className="muted" style={{ fontSize: 12 }}>Snapshot B — top pick</span>
                            <strong style={{ fontSize: 22, color: teamColors[snapB.result?.[0]?.constructor_name] || "var(--accent)" }}>
                              {(topProbB * 100).toFixed(1)}%
                            </strong>
                            <span className="muted" style={{ fontSize: 12 }}>Margin over 2nd: <strong>{(gapB * 100).toFixed(1)}pp</strong></span>
                          </div>
                        </div>
                        <p className="compare-analysis-note">
                          {topProbA > topProbB
                            ? `A is more decisive by ${((topProbA - topProbB) * 100).toFixed(1)}pp.${gapA > gapB ? " Its wider margin over 2nd place also signals a clearer favourite." : ""}`
                            : topProbB > topProbA
                            ? `B is more decisive by ${((topProbB - topProbA) * 100).toFixed(1)}pp.${gapB > gapA ? " Its wider margin over 2nd place also signals a clearer favourite." : ""}`
                            : "Both snapshots show identical top-pick confidence."}
                          {" "}A margin above ~20pp generally indicates high model certainty; below ~10pp the field is too tight to call.
                        </p>
                      </div>

                      {/* Agreement */}
                      <div className="compare-analysis-block">
                        <p className="eyebrow small">Agreement</p>
                        {samePick ? (
                          <p>
                            Both snapshots pick <strong style={{ color: teamColors[snapA.result?.[0]?.constructor_name] || "var(--accent)" }}>{snapA.result?.[0]?.constructor_name}</strong> as champion.
                            The models agree — focus on which snapshot's standings are more realistic as a scenario.
                          </p>
                        ) : (
                          <p>
                            The snapshots <strong>disagree</strong>: A picks{" "}
                            <strong style={{ color: teamColors[snapA.result?.[0]?.constructor_name] || "var(--accent)" }}>{snapA.result?.[0]?.constructor_name}</strong> while B picks{" "}
                            <strong style={{ color: teamColors[snapB.result?.[0]?.constructor_name] || "var(--accent)" }}>{snapB.result?.[0]?.constructor_name}</strong>.
                            {" "}This means the championship outcome is sensitive to the standing changes between these two scenarios — even small shifts in points or wins are enough to flip the model's pick.
                          </p>
                        )}
                      </div>

                    </div>

                    <div className="compare-rows">
                      {rows.map((row) => {
                        const color = teamColors[row.name] || "#9AA0A6";
                        return (
                          <div key={row.name} className="compare-team-row">
                            <div className="compare-team-name">
                              <TeamLogo name={row.name} color={color} size={20} />
                              <span>{row.name}</span>
                            </div>
                            <div className="compare-bars">
                              <div className="compare-bar-wrap">
                                <div className="compare-bar-fill" style={{ width: `${row.probA * 100}%`, background: color, opacity: 0.45 }} />
                              </div>
                              <div className="compare-bar-wrap">
                                <div className="compare-bar-fill" style={{ width: `${row.probB * 100}%`, background: color }} />
                              </div>
                            </div>
                            <div className="compare-probs">
                              <span>{(row.probA * 100).toFixed(1)}%</span>
                              <span>{(row.probB * 100).toFixed(1)}%</span>
                            </div>
                            <div
                              className="compare-delta"
                              style={{ color: row.delta > 0.005 ? "#00a651" : row.delta < -0.005 ? "#e10600" : "var(--muted)" }}
                            >
                              {Math.abs(row.delta) < 0.005 ? "—" : `${row.delta > 0 ? "▲" : "▼"}${(Math.abs(row.delta) * 100).toFixed(1)}%`}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    <p className="panel-note" style={{ marginTop: 12, fontSize: 12 }}>
                      Faded bar = Snapshot A · Solid bar = Snapshot B · Δ = change in predicted probability
                    </p>
                  </div>
                );
              })()}

              {compareIds.length === 1 && (
                <p className="panel-note" style={{ marginBottom: 12 }}>Select one more snapshot to compare.</p>
              )}
              {compareIds.length === 0 && (
                <p className="panel-note" style={{ marginBottom: 12 }}>Click two snapshots to compare them.</p>
              )}

              <div className="saved-grid">
                {savedPredictions.map((snap) => {
                  const isSelected = compareIds.includes(snap.id);
                  const selIdx = compareIds.indexOf(snap.id);
                  const topTeam = snap.result?.[0];
                  const topColor = topTeam ? (teamColors[topTeam.constructor_name] || "#9AA0A6") : "#9AA0A6";
                  return (
                    <article
                      key={snap.id}
                      className={`saved-card ${isSelected ? "saved-card--selected" : ""}`}
                      onClick={() => toggleCompare(snap.id)}
                    >
                      {isSelected && (
                        <div className="saved-card-badge">{selIdx === 0 ? "A" : "B"}</div>
                      )}
                      <div className="saved-card-head">
                        <span className="saved-card-label">{snap.label}</span>
                        <button
                          className="delete-btn"
                          onClick={(e) => { e.stopPropagation(); deleteSnapshot(snap.id); }}
                          title="Delete"
                        >✕</button>
                      </div>
                      <span className="muted" style={{ fontSize: 11 }}>{new Date(snap.timestamp).toLocaleString()}</span>
                      {topTeam && (
                        <div className="saved-card-winner">
                          <TeamLogo name={topTeam.constructor_name} color={topColor} size={24} />
                          <div>
                            <strong style={{ color: topColor, fontSize: 14 }}>{topTeam.constructor_name}</strong>
                            <span className="muted" style={{ fontSize: 12, display: "block" }}>
                              {(topTeam.champion_probability * 100).toFixed(1)}% probability
                            </span>
                          </div>
                        </div>
                      )}
                    </article>
                  );
                })}
              </div>
            </>
          )}
        </section>
      )}

      {/* ══ OVERVIEW ═════════════════════════════════════════════════════════ */}
      {page === "overview" && (
        <section className="panel">

          {/* Header */}
          <div className="panel-header">
            <div>
              <h2>Model Overview</h2>
              <p className="panel-note compact">
                Train the model on historical seasons and evaluate how accurately it predicts constructor championships.
              </p>
            </div>
            <div className="action-row">
              {error && <p className="error">{error}</p>}
              <button className="primary" onClick={runTraining} disabled={loading}>
                {loading ? "Training…" : "Run Model"}
              </button>
            </div>
          </div>

          {/* Explainer cards — always visible */}
          <div className="overview-explainer">
            <article className="compare-card">
              <p className="eyebrow small">How It Works</p>
              <h3>The Model</h3>
              <p className="panel-note">
                A <strong>Random Forest classifier</strong> is trained on every constructor-season from the
                earliest available year up to a configurable cutoff. For each constructor it uses features
                from the <em>previous</em> season to predict whether that team will win the next championship:
              </p>
              <ul className="plain-list">
                <li><strong>Points &amp; Wins (raw)</strong> — the absolute totals accumulated by the end of the prior season.</li>
                <li><strong>Points &amp; Wins (relative)</strong> — each figure as a fraction of the season leader's score, making mid-season snapshots directly comparable to full-season data.</li>
                <li><strong>Championship position</strong> — final finishing position, plus a binary flag for defending champions.</li>
              </ul>
              <p className="panel-note" style={{ marginTop: 10, marginBottom: 0 }}>
                Seasons <em>after</em> the cutoff are held out as a test set — the model never sees them
                during training. Everything on this page measures performance on those unseen seasons only.
              </p>
            </article>

            <article className="compare-card">
              <p className="eyebrow small">Reading the Metrics</p>
              <h3>What the Numbers Mean</h3>
              <ul className="plain-list">
                <li>
                  <strong>Test accuracy</strong> — out of every constructor-season row in the held-out set,
                  the fraction the model correctly labelled "champion" or "not champion". Because roughly
                  1 in 10 teams wins each year, even a trivial model that always predicts "not champion"
                  would score ~90% — so this figure alone is not the most meaningful signal.
                </li>
                <li style={{ marginTop: 8 }}>
                  <strong>Top-1 accuracy</strong> — the more meaningful metric: for each test season, did
                  the model rank the actual champion as its single highest-probability pick? This directly
                  measures real-world usefulness.
                </li>
                <li style={{ marginTop: 8 }}>
                  <strong>Test seasons</strong> — how many complete seasons were held back for evaluation.
                  More test seasons = a more reliable estimate of real-world performance.
                </li>
              </ul>
            </article>
          </div>

          {!result && (
            <div className="overview-empty">
              <p className="muted">Click <strong>Run Model</strong> above to train and evaluate. Metrics and charts will appear here.</p>
            </div>
          )}

          {result && (
            <>
              {/* Stats row */}
              <div className="results-grid" style={{ marginTop: 28 }}>
                <div className="stat">
                  <p>Test accuracy</p>
                  <strong>{result.accuracy.toFixed(3)}</strong>
                  <span className="stat-note">{(result.accuracy * 100).toFixed(1)}% of rows correctly labelled</span>
                </div>
                <div className="stat">
                  <p>Top-1 accuracy</p>
                  <strong>{result.top1_accuracy === null ? "n/a" : result.top1_accuracy.toFixed(3)}</strong>
                  <span className="stat-note">
                    {result.top1_accuracy !== null &&
                      `${result.top1_picks.filter((p) => p.is_true_champion === 1).length} of ${result.test_seasons} seasons correct`}
                  </span>
                </div>
                <div className="stat">
                  <p>Test seasons</p>
                  <strong>{result.test_seasons}</strong>
                  <span className="stat-note">held-out evaluation seasons</span>
                </div>
              </div>

              {/* Chart 1: probability lines */}
              <TeamProbChart allPredictions={result.all_predictions} />

              {/* Chart 2: season cards */}
              <SeasonBreakdown picks={result.top1_picks} allPredictions={result.all_predictions} />
              <UpsetDetector picks={result.top1_picks} allPredictions={result.all_predictions} />
            </>
          )}
        </section>
      )}

      {/* ══ FORECAST ═════════════════════════════════════════════════════════ */}
      {page === "forecast" && (
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Forecast</h2>
              <p className="panel-note compact">Project seasons beyond the dataset.</p>
            </div>
            <div className="action-row">
              <label className="inline-label">
                Years ahead
                <input type="number" min="1" max="10" value={forecastForm.years_ahead}
                  onChange={(e) => setForecastForm((p) => ({ ...p, years_ahead: e.target.value }))} />
              </label>
              <label className="inline-label">
                Carry-forward
                <input type="number" min="0.3" max="0.95" step="0.05" value={forecastForm.carry_forward}
                  onChange={(e) => setForecastForm((p) => ({ ...p, carry_forward: e.target.value }))} />
              </label>
              {forecastError && <p className="error">{forecastError}</p>}
              <button className="primary" onClick={runForecast} disabled={forecastLoading}>
                {forecastLoading ? "Forecasting…" : "Run Forecast"}
              </button>
            </div>
          </div>

          {!forecastResult && <p className="muted">Click Run Forecast to project future seasons.</p>}
          {forecastResult && (
            <>
              <div className="results-grid">
                <div className="stat"><p>Latest data year</p><strong>{forecastResult.source_data_year}</strong></div>
                <div className="stat">
                  <p>Forecast through</p>
                  <strong>{forecastResult.forecast_years[forecastResult.forecast_years.length - 1]?.year}</strong>
                </div>
                <div className="stat">
                  <p>Carry-forward</p>
                  <strong>{Math.round(Number(forecastForm.carry_forward) * 100)}%</strong>
                </div>
              </div>

              <ForecastChart forecastYears={forecastResult.forecast_years} />

              <div className="forecast-layout" style={{ marginTop: 24 }}>
                <article className="compare-card">
                  <p className="eyebrow small">Assumptions</p>
                  <h3>How Future Seasons Are Simulated</h3>
                  <ul className="plain-list">
                    {forecastResult.assumptions.map((item) => <li key={item}>{item}</li>)}
                  </ul>
                </article>
                <RunSummary title="Forecast Uses" presetId={selectedPreset} result={null} />
              </div>

              <div className="forecast-stack">
                {forecastResult.forecast_years.map((fy) => (
                  <article key={fy.year} className="forecast-card">
                    <div className="panel-header">
                      <div>
                        <p className="eyebrow small">Projected Season</p>
                        <h3>{fy.year}</h3>
                      </div>
                      <div className="champion-callout">
                        <span>Champion pick</span>
                        <strong>{fy.champion_pick}</strong>
                      </div>
                    </div>
                    <div className="bar-list">
                      {fy.rows.slice(0, 8).map((row) => {
                        const color = teamColors[row.constructor_name] || "#9AA0A6";
                        return (
                          <div key={`${fy.year}-${row.constructor_id}`} className="bar-row">
                            <div className="bar-label">
                              <TeamLogo name={row.constructor_name} color={color} size={20} />
                              <span>{row.constructor_name}</span>
                            </div>
                            <div className="bar-track">
                              <div className="bar-fill animated" style={{ width: `${Number(row.champion_probability) * 100}%`, background: color }} />
                            </div>
                            <div className="bar-right">
                              <span className="bar-value">{Number(row.champion_probability).toFixed(3)}</span>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </article>
                ))}
              </div>
            </>
          )}
        </section>
      )}

      {/* ══ ABOUT METRICS ════════════════════════════════════════════════════ */}
      {page === "metrics" && (
        <section className="panel">
          <h2>About Metrics</h2>
          <div className="metrics-grid">
            <article className="metric-card">
              <p className="eyebrow small">Accuracy</p>
              <h3>Row-Level Performance</h3>
              <p className="panel-note">How often the model correctly labelled each constructor-season row as champion or not.</p>
            </article>
            <article className="metric-card">
              <p className="eyebrow small">Top-1 Accuracy</p>
              <h3>Season Winner Picking</h3>
              <p className="panel-note">Whether the single highest-probability team in a given season was the real champion.</p>
            </article>
            <article className="metric-card">
              <p className="eyebrow small">Confidence</p>
              <h3>Prediction Certainty</h3>
              <p className="panel-note">
                <strong>High</strong> — top team probability above 60% or large gap to second place.<br />
                <strong>Moderate</strong> — clear favourite but field is closer.<br />
                <strong>Uncertain</strong> — multiple teams tightly bunched.
              </p>
            </article>
          </div>
          {result && (
            <div className="tables single-table">
              <h3>Top-1 Picks</h3>
              <table>
                <thead><tr><th>Year</th><th>Constructor</th><th>Probability</th><th>Correct?</th></tr></thead>
                <tbody>
                  {result.top1_picks.map((row) => (
                    <tr key={`${row.year}-${row.constructor_name}`}>
                      <td>{row.year}</td>
                      <td>{row.constructor_name}</td>
                      <td>{Number(row.predicted_prob).toFixed(3)}</td>
                      <td style={{ color: row.is_true_champion === 1 ? "#00a651" : "#e10600", fontWeight: 600 }}>
                        {row.is_true_champion === 1 ? "Yes" : "No"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      )}

      <footer className="footer">
        <p>Powered by FastAPI + React · Built for exploratory ML analysis</p>
      </footer>
    </div>
  );
}
