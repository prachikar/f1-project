from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np
import pandas as pd
import requests as _requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.load_data import load_f1_data
from src.prepare_features import prepare_features
from src.train_model import train_model, TrainConfig
from src.evaluate import evaluate_model


app = FastAPI(title="F1 Constructor Champion Predictor API")

# Comma-separated list of extra allowed origins, e.g. your Vercel domain.
# Set FRONTEND_ORIGIN=https://f1-project.vercel.app in the backend host's env vars.
_extra_origins = [o.strip() for o in os.environ.get("FRONTEND_ORIGIN", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", *_extra_origins],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TrainRequest(BaseModel):
    train_end_year: int = Field(..., description="Last year included in training set")
    n_estimators: int = Field(300, ge=10, le=2000)
    max_depth: int = Field(6, ge=1, le=64)
    class_weight: Optional[str] = Field("balanced", description="balanced or none")
    random_state: int = 42


class MetricResponse(BaseModel):
    accuracy: float
    confusion_matrix: List[List[int]]
    classification_report: dict
    top1_accuracy: Optional[float]
    top1_picks: List[dict]
    sample_predictions: List[dict]
    all_predictions: List[dict]
    test_seasons: int


class ForecastRequest(TrainRequest):
    years_ahead: int = Field(3, ge=1, le=10)
    carry_forward: float = Field(
        0.7,
        ge=0.3,
        le=0.95,
        description="How much of the previous simulated season to retain",
    )


class ForecastResponse(BaseModel):
    source_data_year: int
    forecast_years: List[dict]
    assumptions: List[str]


class PredictRow(BaseModel):
    constructor_name: str
    points_prev: float
    wins_prev: float
    pos_prev: float


class PredictRequest(BaseModel):
    rows: List[PredictRow]
    n_estimators: int = 300
    max_depth: int = 6
    random_state: int = 42
    class_weight: Optional[str] = "balanced"


def _load_constructor_frame():
    constructors, constructor_standings, races, _results = load_f1_data()
    constructor_map = dict(zip(constructors["constructor_id"], constructors["name"]))

    cs = constructor_standings.copy()
    cs.columns = cs.columns.str.strip().str.lower()
    last_round_per_season = cs.groupby("season")["round"].transform("max")
    cs_final = cs[cs["round"] == last_round_per_season].copy()
    cs_final["position"] = pd.to_numeric(cs_final["position"], errors="coerce")
    cs_final["points"] = pd.to_numeric(cs_final["points"], errors="coerce")
    cs_final["wins"] = pd.to_numeric(cs_final["wins"], errors="coerce")

    return constructors, constructor_map, cs_final, races


def _build_metric_response(pred_df: pd.DataFrame) -> MetricResponse:
    accuracy = float(accuracy_score(pred_df["y_true"], pred_df["y_pred"]))
    cm = confusion_matrix(pred_df["y_true"], pred_df["y_pred"]).tolist()
    report = classification_report(
        pred_df["y_true"], pred_df["y_pred"], output_dict=True, digits=4
    )

    top1_accuracy = None
    top1_picks: List[dict] = []
    if pred_df["y_proba"].notna().all():
        top1 = (
            pred_df.sort_values(["year", "y_proba"], ascending=[True, False])
            .groupby("year", as_index=False)
            .head(1)
        )
        top1_accuracy = float((top1["y_true"] == 1).mean())
        top1_picks = top1[["year", "constructor_name", "y_proba", "y_true"]].rename(
            columns={"y_proba": "predicted_prob", "y_true": "is_true_champion"}
        ).to_dict(orient="records")

    sample_predictions = (
        pred_df.sort_values("y_proba", ascending=False)
        .head(25)
        .to_dict(orient="records")
    )

    all_predictions = pred_df.to_dict(orient="records")
    test_seasons = int(pred_df["year"].nunique())

    return MetricResponse(
        accuracy=accuracy,
        confusion_matrix=cm,
        classification_report=report,
        top1_accuracy=top1_accuracy,
        top1_picks=top1_picks,
        sample_predictions=sample_predictions,
        all_predictions=all_predictions,
        test_seasons=test_seasons,
    )


def _simulate_future_years(
    model,
    latest_state: pd.DataFrame,
    source_year: int,
    years_ahead: int,
    carry_forward: float,
):
    current_state = latest_state.copy()
    total_points = float(current_state["points_prev"].sum())
    total_wins = float(current_state["wins_prev"].sum())
    forecast_years: List[dict] = []

    for step in range(1, years_ahead + 1):
        forecast_year = source_year + step

        max_pts  = current_state["points_prev"].max() or 1
        max_wins = current_state["wins_prev"].max()  or 1
        feature_frame = current_state[["points_prev", "wins_prev", "pos_prev"]].copy()
        feature_frame["points_pct_prev"] = feature_frame["points_prev"] / max_pts
        feature_frame["wins_pct_prev"]   = feature_frame["wins_prev"]   / max_wins
        feature_frame["champ_prev"]      = (feature_frame["pos_prev"] == 1).astype(int)
        probabilities = model.predict_proba(feature_frame)[:, 1]

        forecast_df = current_state[["constructor_id", "constructor_name"]].copy()
        forecast_df["forecast_year"] = forecast_year
        forecast_df["champion_probability"] = probabilities
        forecast_df = forecast_df.sort_values(
            "champion_probability", ascending=False
        ).reset_index(drop=True)
        forecast_df["projected_position"] = np.arange(1, len(forecast_df) + 1)

        normalized = forecast_df["champion_probability"].to_numpy(dtype=float)
        normalized = normalized + 1e-6
        normalized = normalized / normalized.sum()

        projected_points = (
            carry_forward * current_state.set_index("constructor_id").loc[
                forecast_df["constructor_id"], "points_prev"
            ].to_numpy()
            + (1 - carry_forward) * normalized * total_points
        )
        projected_wins = (
            carry_forward * current_state.set_index("constructor_id").loc[
                forecast_df["constructor_id"], "wins_prev"
            ].to_numpy()
            + (1 - carry_forward) * normalized * total_wins
        )

        forecast_df["projected_points_basis"] = projected_points
        forecast_df["projected_wins_basis"] = projected_wins

        forecast_years.append(
            {
                "year": forecast_year,
                "champion_pick": forecast_df.iloc[0]["constructor_name"],
                "rows": forecast_df.to_dict(orient="records"),
            }
        )

        next_state = forecast_df[
            [
                "constructor_id",
                "constructor_name",
                "projected_points_basis",
                "projected_wins_basis",
                "projected_position",
            ]
        ].rename(
            columns={
                "projected_points_basis": "points_prev",
                "projected_wins_basis": "wins_prev",
                "projected_position": "pos_prev",
            }
        )
        current_state = next_state

    return forecast_years


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/seasons")
def get_seasons():
    """Return all available seasons and their max round numbers."""
    constructors, constructor_standings, races, _ = load_f1_data()
    cs = constructor_standings.copy()
    cs.columns = cs.columns.str.strip().str.lower()
    cs["round"] = pd.to_numeric(cs["round"], errors="coerce")
    season_rounds = (
        cs.groupby("season")["round"].max()
        .reset_index()
        .sort_values("season", ascending=False)
    )
    return {
        "seasons": [
            {"season": int(r["season"]), "max_round": int(r["round"])}
            for _, r in season_rounds.iterrows()
        ]
    }


@app.get("/standings/{season}/{round_num}")
def get_standings(season: int, round_num: int):
    """Return constructor standings at a specific season and round."""
    constructors, constructor_standings, races, _ = load_f1_data()
    constructor_map = dict(zip(constructors["constructor_id"], constructors["name"]))
    cs = constructor_standings.copy()
    cs.columns = cs.columns.str.strip().str.lower()
    cs["position"] = pd.to_numeric(cs["position"], errors="coerce")
    cs["points"]   = pd.to_numeric(cs["points"],   errors="coerce")
    cs["wins"]     = pd.to_numeric(cs["wins"],      errors="coerce")
    cs["round"]    = pd.to_numeric(cs["round"],     errors="coerce")

    season_data = cs[cs["season"] == season]
    available_rounds = sorted(season_data["round"].dropna().unique())
    if not available_rounds:
        return {"season": season, "round": round_num, "rows": []}

    actual_round = min(available_rounds, key=lambda r: abs(r - round_num))
    round_data = (
        season_data[season_data["round"] == actual_round]
        .dropna(subset=["position"])
        .sort_values("position")
    )
    return {
        "season": season,
        "round": int(actual_round),
        "rows": [
            {
                "constructor_name": constructor_map.get(r["constructor_id"], r["constructor_id"]),
                "points_prev": float(r["points"] or 0),
                "wins_prev":   float(r["wins"]   or 0),
                "pos_prev":    float(r["position"]),
            }
            for _, r in round_data.iterrows()
        ],
    }


@app.get("/predict/latest")
def predict_latest():
    """Return the most recent season's final standings to pre-populate the predict form."""
    constructors, constructor_map, cs_final, _ = _load_constructor_frame()
    latest_year = int(cs_final["season"].max())
    latest = (
        cs_final[cs_final["season"] == latest_year]
        [["constructor_id", "points", "wins", "position"]]
        .dropna(subset=["position"])
        .sort_values("position")
        .reset_index(drop=True)
    )
    return {
        "season": latest_year,
        "rows": [
            {
                "constructor_name": constructor_map.get(row["constructor_id"], row["constructor_id"]),
                "points_prev": float(row["points"] or 0),
                "wins_prev": float(row["wins"] or 0),
                "pos_prev": float(row["position"]),
            }
            for _, row in latest.iterrows()
        ],
    }


@app.get("/live-standings")
def live_standings():
    """Fetch current season constructor standings from the Jolpica F1 API."""
    url = "https://api.jolpi.ca/ergast/f1/current/constructorStandings.json"
    try:
        resp = _requests.get(url, timeout=8)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return {"error": str(exc), "season": None, "round": None, "rows": []}

    try:
        lists = data["MRData"]["StandingsTable"]["StandingsLists"]
        if not lists:
            return {"error": "Season not started yet", "season": None, "round": None, "rows": []}
        standings = lists[0]
        season    = int(standings["season"])
        round_num = int(standings["round"])
        total = len(standings["ConstructorStandings"])
        rows = [
            {
                "constructor_name": s["Constructor"]["name"],
                "points_prev":      float(s["points"] or 0),
                "wins_prev":        float(s["wins"] or 0),
                "pos_prev":         float(s["position"]) if s.get("position") not in (None, "") else float(total),
            }
            for s in standings["ConstructorStandings"]
        ]
        return {"season": season, "round": round_num, "rows": rows}
    except (KeyError, IndexError, ValueError, TypeError) as exc:
        return {"error": f"Unexpected API response: {exc}", "season": None, "round": None, "rows": []}


@app.get("/constructor-stats")
def constructor_stats(names: str):
    """Return historical performance stats for named constructors."""
    name_list = [n.strip() for n in names.split(",")]
    constructors, constructor_standings, races, _ = load_f1_data()
    constructor_map = dict(zip(constructors["constructor_id"], constructors["name"]))

    cs = constructor_standings.copy()
    cs.columns = cs.columns.str.strip().str.lower()
    cs["position"] = pd.to_numeric(cs["position"], errors="coerce")
    cs["points"]   = pd.to_numeric(cs["points"],   errors="coerce")
    cs["round"]    = pd.to_numeric(cs["round"],     errors="coerce")

    last_round = cs.groupby("season")["round"].transform("max")
    cs_final = cs[cs["round"] == last_round].copy()
    cs_final["constructor_name"] = cs_final["constructor_id"].map(constructor_map)

    result = []
    for name in name_list:
        team = cs_final[cs_final["constructor_name"] == name].sort_values("season")
        if team.empty:
            result.append({
                "constructor_name": name,
                "championships": 0, "seasons": 0,
                "avg_position_5yr": 10.0, "points_trend": 0.0,
            })
            continue

        championships = int((team["position"] == 1).sum())
        seasons       = len(team)
        recent        = team.tail(5)
        avg_pos_5yr   = float(recent["position"].mean())

        # Linear trend of points over last 5 seasons (positive = improving)
        pts = recent["points"].fillna(0).tolist()
        n   = len(pts)
        if n >= 2:
            xm  = (n - 1) / 2
            ym  = sum(pts) / n
            num = sum((i - xm) * (pts[i] - ym) for i in range(n))
            den = sum((i - xm) ** 2 for i in range(n))
            trend = num / den if den else 0.0
        else:
            trend = 0.0

        result.append({
            "constructor_name": name,
            "championships":    championships,
            "seasons":          seasons,
            "avg_position_5yr": avg_pos_5yr,
            "points_trend":     float(trend),
        })

    return {"stats": result}


class RaceBreakdownRequest(BaseModel):
    season: int
    n_estimators: int = 300
    max_depth: int = 6
    random_state: int = 42
    class_weight: Optional[str] = "balanced"


def _fetch_round_standings(season: int, rnd: int) -> tuple[int, list]:
    """Fetch constructor standings for a single round from the Jolpica API (with one retry)."""
    url = f"https://api.jolpi.ca/ergast/f1/{season}/{rnd}/constructorStandings.json"
    for attempt in range(2):
        try:
            resp = _requests.get(url, timeout=12)
            resp.raise_for_status()
            lists = resp.json()["MRData"]["StandingsTable"]["StandingsLists"]
            if not lists:
                return rnd, []
            sl = lists[0]
            total = len(sl["ConstructorStandings"])
            rows = [
                {
                    "constructor_id":   s["Constructor"]["constructorId"],
                    "constructor_name": s["Constructor"]["name"],
                    "points":           float(s["points"] or 0),
                    "wins":             float(s["wins"]   or 0),
                    "position":         float(s["position"]) if s.get("position") not in (None, "") else float(total),
                }
                for s in sl["ConstructorStandings"]
            ]
            return rnd, rows
        except Exception:
            if attempt == 0:
                import time as _time
                _time.sleep(0.5)
    return rnd, []


@app.post("/race-breakdown")
def race_breakdown(req: RaceBreakdownRequest):
    """Return round-by-round championship probability predictions for a season.

    Fetches per-round constructor standings from the Jolpica API concurrently,
    then runs the trained model on each round's standings.
    """
    constructors, constructor_standings, races, _results = load_f1_data()

    X, y, years, _ = prepare_features(constructors, constructor_standings, races)

    config = TrainConfig(
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        random_state=req.random_state,
        class_weight=None if req.class_weight == "none" else req.class_weight,
        n_jobs=-1,
    )
    model = train_model(X, y, config=config)

    # Determine how many rounds the season had from local data
    cs = constructor_standings.copy()
    cs.columns = cs.columns.str.strip().str.lower()
    cs["round"] = pd.to_numeric(cs["round"], errors="coerce")
    season_max_round = cs[cs["season"] == req.season]["round"].max()
    if pd.isna(season_max_round):
        return {"season": req.season, "rounds": [], "error": "Season not found in local data"}
    max_round = int(season_max_round)

    # Fetch all rounds concurrently from Jolpica API
    round_data: dict[int, list] = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(_fetch_round_standings, req.season, r): r for r in range(1, max_round + 1)}
        for future in as_completed(futures):
            rnd, rows = future.result()
            if rows:
                round_data[rnd] = rows

    if not round_data:
        return {"season": req.season, "rounds": [], "error": "No round data available from API"}

    # Build prediction for each round
    breakdown = []
    for rnd in sorted(round_data.keys()):
        standings = round_data[rnd]
        max_pts  = max((s["points"] for s in standings), default=1) or 1
        max_wins = max((s["wins"]   for s in standings), default=1) or 1

        feature_frame = pd.DataFrame([
            {
                "points_prev":     s["points"],
                "wins_prev":       s["wins"],
                "pos_prev":        s["position"],
                "points_pct_prev": s["points"] / max_pts,
                "wins_pct_prev":   s["wins"]   / max_wins,
                "champ_prev":      1.0 if s["position"] == 1 else 0.0,
            }
            for s in standings
        ])

        probs = model.predict_proba(feature_frame)[:, 1]

        preds = sorted(
            [
                {
                    "constructor_name":     s["constructor_name"],
                    "champion_probability": float(p),
                }
                for s, p in zip(standings, probs)
            ],
            key=lambda x: x["champion_probability"],
            reverse=True,
        )

        top_prob = preds[0]["champion_probability"] if preds else 0.0
        gap      = top_prob - (preds[1]["champion_probability"] if len(preds) > 1 else 0.0)
        if top_prob > 0.6 or gap > 0.4:
            conf_label = "High"
        elif top_prob > 0.35 or gap > 0.2:
            conf_label = "Moderate"
        else:
            conf_label = "Uncertain"

        breakdown.append({
            "round":           rnd,
            "top_pick":        preds[0]["constructor_name"] if preds else None,
            "top_probability": top_prob,
            "confidence":      conf_label,
            "predictions":     preds,
        })

    return {"season": req.season, "rounds": breakdown}


@app.post("/predict")
def predict(req: PredictRequest):
    """Train on all available data and predict championship probabilities for given standings."""
    constructors, constructor_standings, races, results = load_f1_data()
    X, y, years, _ = prepare_features(constructors, constructor_standings, races)

    config = TrainConfig(
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        random_state=req.random_state,
        class_weight=None if req.class_weight == "none" else req.class_weight,
        n_jobs=-1,
    )
    model = train_model(X, y, config=config)

    max_pts  = max((r.points_prev for r in req.rows), default=1) or 1
    max_wins = max((r.wins_prev   for r in req.rows), default=1) or 1

    feature_frame = pd.DataFrame([
        {
            "points_prev":     r.points_prev,
            "wins_prev":       r.wins_prev,
            "pos_prev":        r.pos_prev,
            "points_pct_prev": r.points_prev / max_pts,
            "wins_pct_prev":   r.wins_prev / max_wins,
            "champ_prev":      1.0 if r.pos_prev == 1 else 0.0,
        }
        for r in req.rows
    ])
    probabilities = model.predict_proba(feature_frame)[:, 1]

    results_out = [
        {
            "constructor_name": r.constructor_name,
            "points_prev": r.points_prev,
            "wins_prev": r.wins_prev,
            "pos_prev": r.pos_prev,
            "champion_probability": float(prob),
        }
        for r, prob in zip(req.rows, probabilities)
    ]
    results_out.sort(key=lambda x: x["champion_probability"], reverse=True)
    return {"predictions": results_out}


@app.post("/train", response_model=MetricResponse)
def train(req: TrainRequest):
    constructors, constructor_standings, races, _results = load_f1_data()
    X, y, years, constructor_ids = prepare_features(
        constructors, constructor_standings, races
    )

    train_mask = years <= req.train_end_year
    test_mask = years > req.train_end_year

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    years_test = years[test_mask]
    constructor_ids_test = constructor_ids[test_mask]

    config = TrainConfig(
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        random_state=req.random_state,
        class_weight=None if req.class_weight == "none" else req.class_weight,
        n_jobs=-1,
    )
    model = train_model(X_train, y_train, config=config)

    pred_df = evaluate_model(
        model,
        X_test,
        y_test,
        years=years_test,
        constructor_ids=constructor_ids_test,
        show_reports=False,
    )

    constructor_map = dict(zip(constructors["constructor_id"], constructors["name"]))
    pred_df["constructor_name"] = pred_df["constructor_id"].map(constructor_map)

    return _build_metric_response(pred_df)


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    constructors, constructor_map, cs_final, races = _load_constructor_frame()
    X, y, years, _constructor_ids = prepare_features(constructors, cs_final, races)
    train_mask = years <= req.train_end_year

    X_train = X[train_mask]
    y_train = y[train_mask]
    if X_train.empty:
        raise ValueError("No training rows available for the selected train_end_year.")

    config = TrainConfig(
        n_estimators=req.n_estimators,
        max_depth=req.max_depth,
        random_state=req.random_state,
        class_weight=None if req.class_weight == "none" else req.class_weight,
        n_jobs=-1,
    )
    model = train_model(X_train, y_train, config=config)

    # Try to use live standings as the starting state
    live_season = None
    live_round  = None
    latest_state = None

    try:
        resp = _requests.get(
            "https://api.jolpi.ca/ergast/f1/current/constructorStandings.json", timeout=8
        )
        resp.raise_for_status()
        lists = resp.json()["MRData"]["StandingsTable"]["StandingsLists"]
        if lists:
            sl = lists[0]
            live_season = int(sl["season"])
            live_round  = int(sl["round"])
            total = len(sl["ConstructorStandings"])
            rows = [
                {
                    "constructor_id":   s["Constructor"]["constructorId"],
                    "constructor_name": s["Constructor"]["name"],
                    "points_prev":      float(s["points"] or 0),
                    "wins_prev":        float(s["wins"]   or 0),
                    "pos_prev":         float(s["position"]) if s.get("position") not in (None, "") else float(total),
                }
                for s in sl["ConstructorStandings"]
            ]
            latest_state = pd.DataFrame(rows)
    except Exception:
        pass

    using_live = latest_state is not None and not latest_state.empty

    if not using_live:
        # Fall back to historical final standings
        source_data_year = int(cs_final["season"].max())
        latest_state = (
            cs_final[cs_final["season"] == source_data_year][
                ["constructor_id", "points", "wins", "position"]
            ]
            .dropna(subset=["position"])
            .copy()
            .rename(columns={"points": "points_prev", "wins": "wins_prev", "position": "pos_prev"})
        )
        latest_state["constructor_name"] = latest_state["constructor_id"].map(constructor_map)
        latest_state = latest_state.dropna(subset=["constructor_name"]).reset_index(drop=True)
    else:
        source_data_year = live_season

    forecast_years = _simulate_future_years(
        model=model,
        latest_state=latest_state,
        source_year=source_data_year,
        years_ahead=req.years_ahead,
        carry_forward=req.carry_forward,
    )

    if using_live:
        assumptions = [
            f"The model is trained using seasons through {req.train_end_year}.",
            f"Forecast starts from live {live_season} standings after round {live_round}.",
            "Each future season reuses the model's champion probabilities as a ranking signal.",
            f"Projected points and wins keep {req.carry_forward:.0%} of the prior season and blend the rest toward the forecast ranking.",
            "These are scenario projections using carry-forward assumptions, not observed future standings.",
        ]
    else:
        assumptions = [
            f"The model is trained using seasons through {req.train_end_year}.",
            f"Live standings unavailable — forecast starts from the latest full season in your data: {source_data_year}.",
            "Each future season reuses the model's champion probabilities as a ranking signal.",
            f"Projected points and wins keep {req.carry_forward:.0%} of the prior season and blend the rest toward the forecast ranking.",
            "These are scenario projections using carry-forward assumptions, not observed future standings.",
        ]

    return ForecastResponse(
        source_data_year=source_data_year,
        forecast_years=forecast_years,
        assumptions=assumptions,
    )