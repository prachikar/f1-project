from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.load_data import load_f1_data
from src.prepare_features import prepare_features
from src.train_model import train_model, TrainConfig
from src.evaluate import evaluate_model


st.set_page_config(page_title="F1 Champion Predictor", layout="wide", page_icon="🏎️")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Titillium+Web:wght@300;400;600;700;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Titillium Web', sans-serif;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d0d0d 60%, #1a0000 100%);
    border-left: 5px solid #e10600;
    padding: 2rem 2.5rem 1.5rem 2.5rem;
    margin-bottom: 2rem;
    border-radius: 4px;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 900;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #ffffff;
    margin: 0 0 0.3rem 0;
    line-height: 1;
}
.hero h1 span { color: #e10600; }
.hero p {
    color: #aaaaaa;
    font-size: 0.95rem;
    letter-spacing: 1px;
    margin: 0;
}

/* ── Section headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2.5rem 0 1rem 0;
}
.section-header .bar {
    width: 5px;
    height: 32px;
    background: #e10600;
    border-radius: 2px;
    flex-shrink: 0;
}
.section-header h2 {
    font-size: 1.3rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #ffffff;
    margin: 0;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-top: 3px solid #e10600;
    border-radius: 4px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricLabel"] {
    font-size: 0.72rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #888888 !important;
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff !important;
}

/* ── Primary button ── */
.stButton > button[kind="primary"] {
    background: #e10600;
    color: #ffffff;
    border: none;
    border-radius: 2px;
    font-family: 'Titillium Web', sans-serif;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 0.6rem 2rem;
    transition: background 0.2s;
}
.stButton > button[kind="primary"]:hover {
    background: #ff1a00;
    border: none;
}

/* ── Data editor / dataframe ── */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    border: 1px solid #2a2a2a;
    border-radius: 4px;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    background: #111111 !important;
}

/* ── Divider ── */
hr {
    border-color: #2a2a2a !important;
    margin: 2rem 0;
}

/* ── Multiselect tags ── */
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: #e10600 !important;
}

/* ── Winner card ── */
.winner-card {
    background: linear-gradient(135deg, #1a0000, #2a0000);
    border: 1px solid #e10600;
    border-left: 5px solid #e10600;
    border-radius: 4px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.5rem;
}
.winner-card .label {
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #e10600;
    margin-bottom: 4px;
}
.winner-card .name {
    font-size: 2rem;
    font-weight: 900;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #ffffff;
}
.winner-card .prob {
    font-size: 0.9rem;
    color: #aaaaaa;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly dark layout defaults ───────────────────────────────────────────────
CHART_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="#1a1a1a",
    plot_bgcolor="#1a1a1a",
    font_family="Titillium Web, sans-serif",
    font_color="#ffffff",
    title_font_size=14,
    title_font_color="#ffffff",
)

# ── Team colors ───────────────────────────────────────────────────────────────
TEAM_COLORS: dict[str, str] = {
    "Ferrari":          "#DC0000",
    "Red Bull":         "#3671C6",
    "Mercedes":         "#00D2BE",
    "McLaren":          "#FF8700",
    "Williams":         "#37BEDD",
    "Renault":          "#FFF500",
    "Alpine F1 Team":   "#FF87BC",
    "Aston Martin":     "#006F62",
    "Haas F1 Team":     "#B6BABD",
    "AlphaTauri":       "#2B4562",
    "Toro Rosso":       "#C92D4B",
    "Force India":      "#F596C8",
    "Racing Point":     "#F596C8",
    "Sauber":           "#9B0000",
    "Alfa Romeo":       "#900000",
    "BMW Sauber":       "#0067FF",
    "Benetton":         "#00A651",
    "Jordan":           "#FFD700",
    "Toyota":           "#CC1E1E",
    "BAR":              "#ABCDEF",
    "Brawn":            "#FFFB00",
    "Brabham":          "#006EFF",
    "BRM":              "#004225",
    "Tyrrell":          "#005AFF",
    "Ligier":           "#0090D0",
    "Team Lotus":       "#555555",
    "Lotus F1":         "#FFB800",
    "Lotus-Climax":     "#006633",
    "Lotus-Ford":       "#1B3A2D",
    "Cooper-Climax":    "#2D6A2D",
}

# ── Hero banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>F1 <span>Constructor</span> Champion Predictor</h1>
  <p>Machine learning predictions powered by historical Formula 1 data</p>
</div>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return load_f1_data()


@st.cache_data
def build_features(_constructors, _constructor_standings, _races):
    return prepare_features(_constructors, _constructor_standings, _races)


constructors, constructor_standings, races, results = load_data()
X, y, years, constructor_ids = build_features(constructors, constructor_standings, races)
constructor_map = dict(zip(constructors["constructor_id"], constructors["name"]))

# ── Auto-train (fixed config, hidden from user) ───────────────────────────────
@st.cache_resource
def get_trained_model(_X, _y):
    config = TrainConfig(
        n_estimators=300, max_depth=6, random_state=42,
        class_weight="balanced", n_jobs=-1,
    )
    return train_model(_X, _y, config)


model = get_trained_model(X, y)

# ── Shared standings data ─────────────────────────────────────────────────────
cs = constructor_standings.copy()
cs.columns = cs.columns.str.strip().str.lower()
cs["constructor_name"] = cs["constructor_id"].map(constructor_map)
cs["position"] = pd.to_numeric(cs["position"], errors="coerce")
cs["points"]   = pd.to_numeric(cs["points"],   errors="coerce")
cs["wins"]     = pd.to_numeric(cs["wins"],      errors="coerce")

last_round = cs.groupby("season")["round"].transform("max")
cs_final = cs[cs["round"] == last_round].copy()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
  <div class="bar"></div>
  <h2>Predict Next Champion</h2>
</div>
""", unsafe_allow_html=True)

st.caption(
    "Pre-filled with last season's final standings. "
    "Edit any values to explore what-if scenarios, then hit Predict."
)

latest_season = int(cs_final["season"].max())
latest_data = (
    cs_final[cs_final["season"] == latest_season]
    [["constructor_name", "points", "wins", "position"]]
    .dropna(subset=["position"])
    .sort_values("position")
    .reset_index(drop=True)
    .rename(columns={
        "constructor_name": "Constructor",
        "points":           "Points (last season)",
        "wins":             "Wins (last season)",
        "position":         "Final Position (last season)",
    })
)

edited = st.data_editor(
    latest_data,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "Constructor": st.column_config.TextColumn("Constructor", disabled=True),
        "Points (last season)":          st.column_config.NumberColumn(min_value=0, step=1),
        "Wins (last season)":            st.column_config.NumberColumn(min_value=0, step=1),
        "Final Position (last season)":  st.column_config.NumberColumn(min_value=1, step=1),
    },
)

if st.button("Predict Champion", type="primary"):
    X_pred = edited.rename(columns={
        "Points (last season)":         "points_prev",
        "Wins (last season)":           "wins_prev",
        "Final Position (last season)": "pos_prev",
    })[["points_prev", "wins_prev", "pos_prev"]].astype(float)

    if X_pred.isna().any().any():
        st.error("Please fill in all values before predicting.")
    else:
        proba = model.predict_proba(X_pred)[:, 1]
        result_df = edited[["Constructor"]].copy()
        result_df["probability"] = proba
        result_df = result_df.sort_values("probability", ascending=False).reset_index(drop=True)

        winner = result_df.iloc[0]
        st.markdown(f"""
        <div class="winner-card">
          <div class="label">Predicted Champion</div>
          <div class="name">{winner['Constructor']}</div>
          <div class="prob">{winner['probability']:.1%} championship probability</div>
        </div>
        """, unsafe_allow_html=True)

        col_chart, col_table = st.columns([3, 1])
        with col_chart:
            plot_df = result_df.sort_values("probability")
            fig_pred = px.bar(
                plot_df,
                x="probability",
                y="Constructor",
                orientation="h",
                color="Constructor",
                color_discrete_map=TEAM_COLORS,
                text=plot_df["probability"].apply(lambda p: f"{p:.1%}"),
            )
            fig_pred.update_traces(textposition="outside")
            fig_pred.update_layout(
                **CHART_THEME,
                showlegend=False,
                height=max(360, len(result_df) * 36),
                xaxis=dict(tickformat=".0%", title="Championship Probability", gridcolor="#2a2a2a"),
                yaxis=dict(title=""),
                margin=dict(l=0, r=60, t=20, b=20),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

        with col_table:
            st.dataframe(
                result_df[["Constructor", "probability"]]
                .assign(probability=result_df["probability"].apply(lambda p: f"{p:.1%}"))
                .rename(columns={"probability": "Probability"}),
                use_container_width=True,
                hide_index=True,
            )

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — HISTORICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="section-header">
  <div class="bar"></div>
  <h2>Historical Analysis</h2>
</div>
""", unsafe_allow_html=True)

# Chart 1: Championship titles
champions = cs_final[cs_final["position"] == 1]
wins_tally = (
    champions.groupby("constructor_name")
    .size()
    .reset_index(name="championships")
    .sort_values("championships", ascending=True)
    .tail(15)
)
fig_wins = px.bar(
    wins_tally,
    x="championships", y="constructor_name", orientation="h",
    title="CONSTRUCTOR CHAMPIONSHIP TITLES — ALL TIME TOP 15",
    labels={"championships": "Titles", "constructor_name": ""},
    color="constructor_name",
    color_discrete_map=TEAM_COLORS,
)
fig_wins.update_layout(**CHART_THEME, showlegend=False, height=450,
    xaxis=dict(gridcolor="#2a2a2a", title="Championships"),
    yaxis=dict(gridcolor="#2a2a2a"),
)
st.plotly_chart(fig_wins, use_container_width=True)

# Chart 2: Points trend
st.markdown("##### SEASON POINTS OVER TIME")
all_constructors_sorted = (
    cs_final.groupby("constructor_name")["points"].sum()
    .sort_values(ascending=False).index.tolist()
)
selected_teams = st.multiselect(
    "Select constructors",
    options=all_constructors_sorted,
    default=all_constructors_sorted[:6],
)
if selected_teams:
    fig_points = px.line(
        cs_final[cs_final["constructor_name"].isin(selected_teams)].sort_values("season"),
        x="season", y="points", color="constructor_name",
        markers=True,
        color_discrete_map=TEAM_COLORS,
        labels={"season": "Season", "points": "Points", "constructor_name": ""},
    )
    fig_points.update_layout(**CHART_THEME, height=420,
        xaxis=dict(gridcolor="#2a2a2a"),
        yaxis=dict(gridcolor="#2a2a2a"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_points, use_container_width=True)

# Chart 3: Standings heatmap
st.markdown("##### CONSTRUCTOR STANDINGS HEATMAP")
top_constructors = (
    cs_final.groupby("constructor_name")["points"].sum()
    .sort_values(ascending=False).head(20).index.tolist()
)
pivot = (
    cs_final[cs_final["constructor_name"].isin(top_constructors)]
    .pivot_table(index="constructor_name", columns="season", values="position", aggfunc="min")
)
pivot = pivot.loc[pivot.mean(axis=1).sort_values().index]

fig_heat = px.imshow(
    pivot,
    color_continuous_scale="RdYlGn_r",
    aspect="auto",
    labels={"x": "Season", "y": "", "color": "Position"},
)
fig_heat.update_layout(**CHART_THEME, height=520,
    coloraxis_colorbar=dict(title="Position", tickfont=dict(color="#ffffff")),
)
st.plotly_chart(fig_heat, use_container_width=True)

# Chart 4: Race wins stacked area
st.markdown("##### RACE WINS PER SEASON")
results_clean = results.copy()
results_clean.columns = results_clean.columns.str.strip().str.lower()
results_clean["position"] = pd.to_numeric(results_clean["position"], errors="coerce")

races_clean = races.copy()
races_clean.columns = races_clean.columns.str.strip().str.lower()

race_wins = (
    results_clean[results_clean["position"] == 1]
    .merge(races_clean[["race_id", "season"]], on="race_id", how="left")
)
race_wins["constructor_name"] = race_wins["constructor_id"].map(constructor_map)

wins_per_season = (
    race_wins[race_wins["constructor_name"].isin(top_constructors)]
    .groupby(["season", "constructor_name"]).size()
    .reset_index(name="wins")
)
fig_area = px.area(
    wins_per_season.sort_values("season"),
    x="season", y="wins", color="constructor_name",
    color_discrete_map=TEAM_COLORS,
    labels={"season": "Season", "wins": "Race Wins", "constructor_name": ""},
)
fig_area.update_layout(**CHART_THEME, height=420,
    xaxis=dict(gridcolor="#2a2a2a"),
    yaxis=dict(gridcolor="#2a2a2a"),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig_area, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — MODEL PERFORMANCE (collapsible)
# ═══════════════════════════════════════════════════════════════════════════════
with st.expander("Model Performance", expanded=False):
    train_mask = years <= 2015
    test_mask  = years > 2015

    eval_model = get_trained_model(X[train_mask], y[train_mask])
    pred_df = evaluate_model(
        eval_model, X[test_mask], y[test_mask],
        years=years[test_mask], constructor_ids=constructor_ids[test_mask],
        show_reports=False,
    )
    pred_df["constructor_name"] = pred_df["constructor_id"].map(constructor_map)

    accuracy = accuracy_score(pred_df["y_true"], pred_df["y_pred"])
    cm = confusion_matrix(pred_df["y_true"], pred_df["y_pred"])
    report = classification_report(pred_df["y_true"], pred_df["y_pred"], output_dict=True, digits=4)

    col1, col2, col3 = st.columns(3)
    col1.metric("Test Accuracy",  f"{accuracy:.3f}")
    col2.metric("Train Cutoff",   "2015")
    col3.metric("Test Seasons",   str(len(pred_df["year"].unique())))

    # Feature importance
    feat_imp = pd.DataFrame({
        "feature":    X.columns.tolist(),
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True)

    fig_imp = px.bar(
        feat_imp, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="reds",
        labels={"importance": "Importance", "feature": ""},
    )
    fig_imp.update_layout(**CHART_THEME, coloraxis_showscale=False, height=260,
        xaxis=dict(gridcolor="#2a2a2a"), yaxis=dict(gridcolor="#2a2a2a"),
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    # Per-year picks
    if pred_df["y_proba"].notna().all():
        top1 = (
            pred_df.sort_values(["year", "y_proba"], ascending=[True, False])
            .groupby("year", as_index=False).head(1)
        )
        top1["correct"] = top1["y_true"] == 1
        st.metric("Top-1 Per-Year Accuracy", f"{(top1['y_true'] == 1).mean():.3f}")

        fig_picks = px.bar(
            top1, x="year", y="y_proba",
            color="correct",
            color_discrete_map={True: "#00c853", False: "#e10600"},
            hover_data=["constructor_name"],
            labels={"y_proba": "Predicted Probability", "year": "Season", "correct": "Correct"},
        )
        fig_picks.update_layout(**CHART_THEME, height=340, showlegend=True,
            xaxis=dict(gridcolor="#2a2a2a"), yaxis=dict(gridcolor="#2a2a2a"),
        )
        st.plotly_chart(fig_picks, use_container_width=True)

        # Probability heatmap
        pivot_proba = pred_df.pivot_table(index="constructor_name", columns="year", values="y_proba")
        pivot_proba = pivot_proba.dropna(how="all")
        pivot_proba = pivot_proba.loc[pivot_proba.mean(axis=1).sort_values(ascending=False).index]

        fig_prob_heat = px.imshow(
            pivot_proba,
            color_continuous_scale="reds",
            aspect="auto", zmin=0, zmax=1,
            labels={"x": "Year", "y": "", "color": "Probability"},
        )
        fig_prob_heat.update_layout(**CHART_THEME, height=460)
        st.plotly_chart(fig_prob_heat, use_container_width=True)

    # Confusion matrix
    fig_cm = px.imshow(
        cm, text_auto=True,
        x=["Predicted 0", "Predicted 1"],
        y=["Actual 0", "Actual 1"],
        color_continuous_scale="reds",
    )
    fig_cm.update_layout(**CHART_THEME, height=300, coloraxis_showscale=False)
    st.plotly_chart(fig_cm, use_container_width=True)

    st.dataframe(pd.DataFrame(report).T, use_container_width=True)
