// Deterministic demonstration of the race condition at frontend/src/App.jsx:875-887.
//
// This is the frontend analog of the Therac-25 root cause: two concurrent,
// unsynchronized async tasks write to the same shared state, and whichever
// one *finishes* last wins -- not whichever one was *started* last. Therac-25
// raced a keyboard-handler task against a treatment-monitor task over a
// shared variable with no lock; here, two `/constructor-stats` fetches
// (fired by the same useEffect, once per predictResult change) race to call
// setAdjustedResult with no cancellation and no "is this still current?"
// check.
//
// Run with: node tests/race_condition_demo.mjs

// ---------------------------------------------------------------------------
// computeAdjusted mirrors frontend/src/App.jsx:81-127 exactly (pure function,
// no shared state of its own -- the bug is entirely in how/when it's called).
function computeAdjusted(predictions, stats) {
  const ALPHA = 0.38;
  const statsMap = Object.fromEntries(stats.map((s) => [s.constructor_name, s]));

  const enriched = predictions.map((p) => {
    const s = statsMap[p.constructor_name] ?? { championships: 0, seasons: 1, avg_position_5yr: 10, points_trend: 0 };
    return { ...p, _s: s };
  });

  const champRates = enriched.map((p) => p._s.championships / Math.max(p._s.seasons, 1));
  const maxCR = Math.max(...champRates, 0.001);
  const normCR = champRates.map((r) => r / maxCR);

  const forms = enriched.map((p) => 1 / Math.max(p._s.avg_position_5yr, 1));
  const maxF = Math.max(...forms, 0.001);
  const normF = forms.map((f) => f / maxF);

  const trends = enriched.map((p) => p._s.points_trend);
  const minT = Math.min(...trends), maxT = Math.max(...trends);
  const normT = trends.map((t) => (maxT === minT ? 0.5 : (t - minT) / (maxT - minT)));

  const histScores = enriched.map((_, i) => 0.50 * normCR[i] + 0.35 * normF[i] + 0.15 * normT[i]);
  const maxH = Math.max(...histScores, 0.001);
  const normH = histScores.map((h) => h / maxH);

  const raw = enriched.map((p, i) => ({
    constructor_name: p.constructor_name,
    _adjusted: p.champion_probability * (1 - ALPHA * (1 - normH[i])),
  }));
  const totalAdj = raw.reduce((s, r) => s + r._adjusted, 0) || 1;

  return raw
    .map((r) => ({ ...r, champion_probability: r._adjusted / totalAdj }))
    .sort((a, b) => b.champion_probability - a.champion_probability);
}

// Fake backend for /constructor-stats -- responseDelayMs simulates real
// network/server jitter (a larger constructor list, a slow moment, etc).
function fakeConstructorStatsCall(statsForRun, responseDelayMs) {
  return new Promise((resolve) => setTimeout(() => resolve(statsForRun), responseDelayMs));
}

// ---------------------------------------------------------------------------
// BUGGY behavior: a faithful port of the real useEffect at App.jsx:875-887.
// No AbortController, no sequence id -- whichever promise resolves last wins.
function buggyEffect(uiState, predictResult, stats, responseDelayMs) {
  fakeConstructorStatsCall(stats, responseDelayMs).then((fetchedStats) => {
    uiState.adjustedResult = computeAdjusted(predictResult, fetchedStats);
    uiState.adjustedFor = predictResult.__label;
  });
}

// FIXED behavior: same shape, but guarded with a monotonically increasing
// request id -- a stale response is discarded instead of overwriting state.
function fixedEffect(uiState, predictResult, stats, responseDelayMs, requestIdBox) {
  const myRequestId = ++requestIdBox.current;
  fakeConstructorStatsCall(stats, responseDelayMs).then((fetchedStats) => {
    if (myRequestId !== requestIdBox.current) {
      console.log(`  [fixed]  discarded stale response for ${predictResult.__label} (request #${myRequestId}, latest is #${requestIdBox.current})`);
      return;
    }
    uiState.adjustedResult = computeAdjusted(predictResult, fetchedStats);
    uiState.adjustedFor = predictResult.__label;
  });
}

// ---------------------------------------------------------------------------
// Two "predictions" a user produced seconds apart by clicking Predict twice
// (button re-enables the moment /predict resolves, well before the
// background /constructor-stats fetch for the *first* run has completed).
const predictionA = Object.assign(
  [
    { constructor_name: "Ferrari", champion_probability: 0.30 },
    { constructor_name: "McLaren", champion_probability: 0.28 },
  ],
  { __label: "A (first Predict click)" }
);
const statsA = [
  { constructor_name: "Ferrari", championships: 16, seasons: 74, avg_position_5yr: 3.1, points_trend: 12 },
  { constructor_name: "McLaren", championships: 8, seasons: 58, avg_position_5yr: 4.4, points_trend: -3 },
];

const predictionB = Object.assign(
  [
    { constructor_name: "Red Bull", champion_probability: 0.33 },
    { constructor_name: "Mercedes", champion_probability: 0.31 },
  ],
  { __label: "B (second Predict click, moments later)" }
);
const statsB = [
  { constructor_name: "Red Bull", championships: 6, seasons: 20, avg_position_5yr: 1.8, points_trend: 20 },
  { constructor_name: "Mercedes", championships: 8, seasons: 58, avg_position_5yr: 2.4, points_trend: 5 },
];

async function main() {
  console.log("=== BUGGY: current App.jsx behavior ===");
  const buggyState = { adjustedResult: null, adjustedFor: null };

  // predictResult A settles first -> effect for A fires, its fetch takes
  // 120ms (e.g. it happened to hit a slower moment on the API).
  buggyEffect(buggyState, predictionA, statsA, 120);

  // 30ms later the user has already clicked Predict again -> predictResult
  // is now B -> a second effect instance fires, its fetch takes only 20ms.
  await sleep(30);
  buggyEffect(buggyState, predictionB, statsB, 20);

  await sleep(200); // let both fake network calls finish

  console.log(`  latest predictResult on screen: ${predictionB.__label}`);
  console.log(`  adjustedResult actually shown for: ${buggyState.adjustedFor}`);
  const buggyMismatch = buggyState.adjustedFor !== predictionB.__label;
  console.log(`  MISMATCH (stale data silently displayed): ${buggyMismatch}`);
  if (buggyMismatch) {
    console.log(`  -> UI shows predictResult[0] = ${predictionB[0].constructor_name}`);
    console.log(`  -> UI shows adjustedResult[0] = ${buggyState.adjustedResult[0].constructor_name} (computed from the OLD prediction)`);
    console.log(`  -> the "differs from model pick" badge (App.jsx:1110) now compares two unrelated runs`);
  }

  console.log("\n=== FIXED: with a request-sequence guard ===");
  const fixedState = { adjustedResult: null, adjustedFor: null };
  const requestIdBox = { current: 0 };

  fixedEffect(fixedState, predictionA, statsA, 120, requestIdBox);
  await sleep(30);
  fixedEffect(fixedState, predictionB, statsB, 20, requestIdBox);

  await sleep(200);

  console.log(`  latest predictResult on screen: ${predictionB.__label}`);
  console.log(`  adjustedResult actually shown for: ${fixedState.adjustedFor}`);
  console.log(`  MISMATCH: ${fixedState.adjustedFor !== predictionB.__label}`);

  if (buggyMismatch && fixedState.adjustedFor === predictionB.__label) {
    console.log("\nRace condition confirmed and fix verified.");
    process.exitCode = 0;
  } else {
    console.log("\nDid not reproduce as expected -- inspect timings above.");
    process.exitCode = 1;
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

main();
