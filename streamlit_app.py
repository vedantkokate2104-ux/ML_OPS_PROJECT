"""
streamlit_app.py
EventOracle / Project Apex – Streamlit Dashboard

Event-Driven Indian Financial Market Prediction System.
Reads pipeline outputs from MongoDB Atlas and reuses the existing
modules (modeling, hedging, market_agent, news_pipeline) directly –
no separate Flask server is needed.

Usage:
    streamlit run streamlit_app.py

Configuration (env vars, .env, or Streamlit Secrets):
    MONGO_URI        – MongoDB Atlas connection string (required)
    GEMINI_API_KEY   – Google Gemini key for the AI Market Agent (optional)
"""

import os
import pickle

import streamlit as st

# ─────────────────────────────────────────────
# SECRETS → ENV  (must run before project modules read os.getenv at import)
# ─────────────────────────────────────────────

st.set_page_config(page_title="EventOracle – Market Intelligence", page_icon="⚡", layout="wide")

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except Exception:
    pass

try:
    for _key in ("MONGO_URI", "GEMINI_API_KEY", "FRED_API_KEY", "DATA_GOV_KEY", "GNEWS_API_KEY"):
        if _key in st.secrets and not os.getenv(_key):
            os.environ[_key] = str(st.secrets[_key])
except Exception:
    pass  # no secrets.toml – fall back to environment variables

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import app_metrics
import data_access as da
from data_access import KNOWN_ASSETS

app_metrics.start_server()

# ─────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────

GREEN, RED, AMBER, BLUE, CYAN, VIOLET = "#10b981", "#f43f5e", "#f59e0b", "#3b82f6", "#06b6d4", "#8b5cf6"
PLOT_LAYOUT = dict(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                   margin=dict(l=10, r=10, t=40, b=10), font=dict(family="Inter, sans-serif"))
SENTIMENT_COLORS = {"positive": GREEN, "negative": RED, "neutral": "#94a3b8"}
DIRECTION_LABEL = {1: "BUY", 0: "HOLD", -1: "SELL"}
DIRECTION_COLOR = {1: GREEN, 0: AMBER, -1: RED}

st.markdown("""
<style>
  .stApp { background: radial-gradient(circle at 20% 0%, #0c1424 0%, #06080d 55%); }
  [data-testid="stSidebar"] { background: #0a0e16; border-right: 1px solid #1e293b; }
  [data-testid="stMetric"] { background: rgba(17,24,39,0.75); border: 1px solid #1e293b;
                             border-radius: 10px; padding: 12px 14px; }
  [data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.35rem; }
  h1, h2, h3 { letter-spacing: -0.02em; }
  .signal { display:inline-block; padding:10px 22px; border-radius:8px; font-weight:800;
            font-size:1.6rem; font-family:'JetBrains Mono', monospace; }
  .muted { color:#94a3b8; font-size:0.85rem; }
</style>
""", unsafe_allow_html=True)


def styled(fig, height=380, title=None):
    fig.update_layout(**PLOT_LAYOUT, height=height, title=title)
    return fig


# ─────────────────────────────────────────────
# DATA LAYER (cached)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Connecting to MongoDB Atlas…")
def get_mongo():
    with app_metrics.track("mongo_connect"):
        return da.get_client()


def mongo_or_none():
    try:
        client = get_mongo()
        app_metrics.mongo_status(True)
        return client
    except Exception as e:
        app_metrics.mongo_status(False)
        st.session_state["mongo_error"] = str(e)
        return None


def _cached_query(op: str, fn, *args, **kwargs) -> pd.DataFrame:
    client = mongo_or_none()
    if client is None:
        return pd.DataFrame()
    try:
        with app_metrics.track(op):
            return fn(client, *args, **kwargs)
    except Exception as e:
        st.warning(f"Could not load {op}: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def q_latest_prices():
    return _cached_query("latest_prices", da.latest_prices)


@st.cache_data(ttl=600, show_spinner=False)
def q_prices(asset, days=None):
    return _cached_query("prices", da.prices, asset, days=days)


@st.cache_data(ttl=600, show_spinner=False)
def q_rows(collection, asset, limit=30):
    return _cached_query(collection, da.latest_rows, collection, asset, limit=limit)


@st.cache_data(ttl=600, show_spinner=False)
def q_news(asset=None, limit=200):
    return _cached_query("news", da.news, asset=asset, limit=limit)


@st.cache_data(ttl=1800, show_spinner=False)
def q_events():
    return _cached_query("events", da.events)


@st.cache_data(ttl=1800, show_spinner=False)
def q_flows():
    return _cached_query("flows", da.flows)


@st.cache_data(ttl=1800, show_spinner=False)
def q_counts():
    client = mongo_or_none()
    return da.collection_counts(client) if client else {}


@st.cache_resource(show_spinner="Loading LightGBM models…")
def get_models():
    import modeling
    clf, reg, selected, idx_to_label = modeling.load_models()
    with open(modeling.LABEL_ENCODER_PATH, "rb") as f:
        enc = pickle.load(f)
    return clf, reg, selected, idx_to_label, enc


@st.cache_data(ttl=1800, show_spinner=False)
def live_inference(asset: str) -> pd.DataFrame:
    """
    Run the trained two-stage LightGBM models on the latest feature rows,
    reusing the exact feature engineering from modeling.py (merge → cross-asset
    → regimes → lagged feature matrix → predict).
    """
    import modeling
    client = get_mongo()
    with app_metrics.track("live_inference"):
        feat = modeling.load_collection(client, modeling.FEATURES_COL, asset=asset)
        if feat.empty:
            return pd.DataFrame()
        social = modeling.load_collection(client, modeling.SOCIAL_COL, asset=asset)
        df = modeling.merge_sources(feat, social)

        # Cross-asset return signals only need the source assets' returns
        sources = [a for a in modeling.CROSS_ASSET_MAP if a != asset]
        src_docs = list(client[modeling.DB_NAME][modeling.FEATURES_COL].find(
            {"asset": {"$in": sources}}, {"_id": 0, "timestamp": 1, "asset": 1, "returns": 1}))
        if src_docs:
            src = pd.DataFrame(src_docs)
            src["timestamp"] = pd.to_datetime(src["timestamp"], utc=True).dt.tz_convert(modeling.IST)
            df = modeling.add_cross_asset_features(pd.concat([df, src], ignore_index=True))
            df = df[df["asset"] == asset].copy()

        df = modeling.add_regime_labels(df)
        _, _, _, _, enc = get_models()
        df, feature_cols, _ = modeling.build_feature_matrix(df, encoder=enc, fit_encoder=False)
        df = modeling.predict(df, feature_cols)

        clf, _, selected, idx_to_label, _ = get_models()
        proba = clf.predict_proba(df[selected].fillna(0).astype(float))
        for idx, label in idx_to_label.items():
            df[f"prob_{DIRECTION_LABEL.get(int(label), label)}"] = proba[:, int(idx)]
        keep = ["timestamp", "close", "returns", "predicted_direction", "predicted_magnitude",
                "confidence_score", "regime_label"] + [c for c in df.columns if c.startswith("prob_")]
        return df[[c for c in keep if c in df.columns]].sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=600, show_spinner=False)
def q_hedging(asset):
    from hedging import get_hedging_strategy
    with app_metrics.track("hedging"):
        return get_hedging_strategy(asset)


@st.cache_resource(show_spinner=False)
def get_vader():
    from news_pipeline import _vader_sentiment
    return _vader_sentiment


def no_data(msg="No data available for this selection yet."):
    st.info(f"ℹ️ {msg}")


def fmt(v, spec=",.2f", fallback="—"):
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return fallback
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────

def page_dashboard():
    st.title("⚡ Executive Dashboard")
    st.caption("Event-driven intelligence across Indian equities, currency, commodities and crypto.")

    with st.spinner("Loading market overview…"):
        latest = q_latest_prices()
    if latest.empty or "close" not in latest:
        no_data("Price data unavailable – run market_pipeline.py or check MONGO_URI.")
        return

    st.subheader("Tracked Assets – Latest Prices")
    cols = st.columns(5)
    for i, row in latest.reset_index(drop=True).iterrows():
        ret = row.get("returns")
        cols[i % 5].metric(row["asset"], fmt(row.get("close")),
                           None if pd.isna(ret) else f"{ret * 100:+.2f}%")
    ts = latest["timestamp"].max() if "timestamp" in latest else None
    if ts is not None and not pd.isna(ts):
        st.markdown(f"<span class='muted'>Last market timestamp: {ts:%d %b %Y %H:%M} IST</span>",
                    unsafe_allow_html=True)

    st.subheader("Market Movements")
    c1, c2 = st.columns([2, 1])
    window = c2.select_slider("Window", options=[30, 90, 180, 365], value=90, format_func=lambda d: f"{d}d")
    selected = c2.multiselect("Assets", KNOWN_ASSETS, default=["NIFTY", "BANKNIFTY", "GOLD", "BITCOIN", "BRENT"])
    frames = []
    with st.spinner("Loading price history…"):
        for a in selected:
            p = q_prices(a, days=window)
            if not p.empty:
                frames.append(p.assign(rebased=p["close"] / p["close"].iloc[0] * 100)[["timestamp", "asset", "rebased"]])
    if frames:
        fig = px.line(pd.concat(frames), x="timestamp", y="rebased", color="asset")
        c1.plotly_chart(styled(fig, 400, f"Rebased performance (start = 100), last {window} days"),
                        width="stretch")
    else:
        c1.info("Select at least one asset with price history.")

    st.subheader("Key Indicators")
    k = st.columns(4)
    fl = q_flows()
    if not fl.empty:
        last = fl.iloc[-1]
        k[0].metric("FII Net (₹ Cr)", fmt(last.get("fii_net"), ",.0f"))
        k[1].metric("DII Net (₹ Cr)", fmt(last.get("dii_net"), ",.0f"))
    nw = q_news(limit=300)
    if not nw.empty and "sentiment_score" in nw:
        k[2].metric("Avg News Sentiment (last 300)", fmt(nw["sentiment_score"].mean(), "+.3f"))
    k[3].metric("MongoDB Documents", fmt(sum(q_counts().values()), ",.0f"))

    if not latest.empty and "rsi" in latest:
        ind = latest[["asset", "close", "returns", "rsi", "rolling_volatility", "atr", "volume_zscore"]].copy()
        ind["returns"] = ind["returns"] * 100
        st.dataframe(ind.rename(columns={"returns": "return %"}).round(3), width="stretch", hide_index=True)


def page_prediction():
    st.title("🎯 Market Prediction")
    st.caption("Two-stage LightGBM: Stage 1 classifies event impact direction, Stage 2 regresses magnitude.")
    asset = st.selectbox("Asset", KNOWN_ASSETS, key="pred_asset")

    tab_live, tab_hist, tab_agent = st.tabs(["Live model inference", "Stored prediction history", "AI Market Agent"])

    with tab_live:
        try:
            with st.spinner(f"Running LightGBM inference for {asset}…"):
                live = live_inference(asset)
        except FileNotFoundError as e:
            st.error(f"Model files missing: {e}")
            live = pd.DataFrame()
        except Exception as e:
            st.error(f"Live inference failed: {e}")
            live = pd.DataFrame()

        if live.empty:
            no_data("No feature rows for this asset – run the pipelines first.")
        else:
            last = live.iloc[-1]
            d = int(last["predicted_direction"])
            c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
            c1.markdown(f"<div class='muted'>Signal for next session</div>"
                        f"<div class='signal' style='background:{DIRECTION_COLOR[d]}22;color:{DIRECTION_COLOR[d]};"
                        f"border:1px solid {DIRECTION_COLOR[d]}'>{DIRECTION_LABEL[d]}</div>", unsafe_allow_html=True)
            c2.metric("Confidence", f"{last['confidence_score'] * 100:.1f}%")
            c3.metric("Expected |move|", f"{last['predicted_magnitude'] * 100:.2f}%")
            c4.metric("Regime", {0: "Low vol", 1: "Mid vol", 2: "High vol"}.get(int(last.get("regime_label", -1)), "—"))
            st.caption(f"As of {last['timestamp']:%d %b %Y} · regime from KMeans on volatility/return z-score")

            prob_cols = [c for c in ("prob_SELL", "prob_HOLD", "prob_BUY") if c in live]
            if prob_cols:
                pc1, pc2 = st.columns([1, 2])
                probs = pd.DataFrame({"class": [c[5:] for c in prob_cols], "p": [last[c] for c in prob_cols]})
                fig = px.bar(probs, x="class", y="p", color="class",
                             color_discrete_map={"SELL": RED, "HOLD": AMBER, "BUY": GREEN})
                fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
                pc1.plotly_chart(styled(fig, 320, "Class probabilities"), width="stretch")

                recent = live.tail(120)
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=recent["timestamp"], y=recent["close"], name="Close",
                                         line=dict(color=CYAN)))
                for dv, name in DIRECTION_LABEL.items():
                    pts = recent[recent["predicted_direction"] == dv]
                    if dv != 0 and not pts.empty:
                        fig.add_trace(go.Scatter(x=pts["timestamp"], y=pts["close"], mode="markers", name=name,
                                                 marker=dict(color=DIRECTION_COLOR[dv], size=8,
                                                             symbol="triangle-up" if dv == 1 else "triangle-down")))
                pc2.plotly_chart(styled(fig, 320, "Price with model signals (last 120 sessions)"),
                                 width="stretch")

            with st.expander("Raw inference output"):
                st.dataframe(live.tail(30).iloc[::-1], width="stretch", hide_index=True)

    with tab_hist:
        with st.spinner("Loading stored predictions…"):
            hist = q_rows("predictions", asset, limit=250)
        if hist.empty:
            no_data("No predictions stored – run `python modeling.py --predict`.")
        else:
            hist["signal"] = hist["predicted_direction"].map(DIRECTION_LABEL)
            fig = px.scatter(hist, x="timestamp", y="confidence_score", color="signal",
                             color_discrete_map={"BUY": GREEN, "SELL": RED, "HOLD": AMBER},
                             size=hist["predicted_magnitude"].fillna(0).clip(lower=1e-4))
            st.plotly_chart(styled(fig, 360, "Stored LightGBM predictions (size = magnitude)"),
                            width="stretch")
            if "strategy_return" in hist and hist["strategy_return"].notna().any():
                eq = (1 + hist["strategy_return"].fillna(0)).cumprod()
                fig = px.area(x=hist["timestamp"], y=eq, labels={"x": "", "y": "Equity"})
                fig.update_traces(line_color=VIOLET)
                st.plotly_chart(styled(fig, 300, "Backtest equity curve (0.1% one-way cost)"),
                                width="stretch")
            if "model_agreement" in hist:
                agree = hist["model_agreement"].dropna()
                if len(agree):
                    st.metric("LightGBM ↔ LSTM agreement", f"{agree.astype(bool).mean() * 100:.1f}%")
            cols = [c for c in ["timestamp", "signal", "predicted_magnitude", "confidence_score", "regime",
                                "lstm_predicted_direction", "lstm_confidence", "agreement_status",
                                "strategy_return"] if c in hist]
            st.dataframe(hist[cols].iloc[::-1], width="stretch", hide_index=True)

    with tab_agent:
        st.write("Gemini 2.5 Flash synthesises the ML prediction, social anomaly and technical features "
                 "into a trading recommendation (uses `market_agent.py`).")
        if not os.getenv("GEMINI_API_KEY"):
            st.warning("GEMINI_API_KEY is not configured – add it to `.env` or Streamlit Secrets to enable the agent.")
        elif st.button(f"Generate AI report for {asset}", type="primary"):
            with st.spinner("Querying Gemini…"):
                try:
                    from market_agent import generate_market_report
                    with app_metrics.track("gemini_agent"):
                        st.markdown(generate_market_report(asset))
                except Exception as e:
                    st.error(f"Agent error: {e}")


def page_events():
    st.title("🏛️ Event Intelligence")
    st.caption("Macro events (RBI, Fed, CPI, GDP, IIP) and FII/DII institutional flows.")

    with st.spinner("Loading events…"):
        ev = q_events()
    if ev.empty:
        no_data("Events collection empty – run events_pipeline.py.")
    else:
        ev["event_name"] = ev["event_name"].astype(str)
        c1, c2 = st.columns(2)
        countries = sorted(ev["country"].dropna().unique()) if "country" in ev else []
        pick_c = c1.multiselect("Country", countries, default=countries)
        names = sorted(ev["event_name"].unique())
        pick_e = c2.multiselect("Event", names, default=names)
        view = ev[ev["event_name"].isin(pick_e) & (ev["country"].isin(pick_c) if pick_c else True)]

        policy = view[view["event_name"].str.contains("rbi|fed|repo|rate", case=False)]
        macro = view[view["event_name"].str.contains("cpi|gdp|iip|inflation", case=False)]

        t1, t2, t3 = st.tabs(["Central bank (RBI / Fed)", "Inflation & growth", "All events"])
        with t1:
            if policy.empty:
                no_data("No RBI/Fed events in selection.")
            else:
                fig = px.line(policy.sort_values("timestamp"), x="timestamp", y="actual",
                              color="event_name", markers=True, line_shape="hv")
                st.plotly_chart(styled(fig, 380, "Policy rates"), width="stretch")
        with t2:
            if macro.empty:
                no_data("No CPI/GDP/IIP events in selection.")
            else:
                macro = macro.assign(series=macro["event_name"] + " (" + macro["country"].fillna("") + ")")
                fig = px.line(macro.sort_values("timestamp"), x="timestamp", y="actual", color="series",
                              markers=True, facet_row="series", height=160 * macro["series"].nunique() + 80)
                fig.update_yaxes(matches=None)
                fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                st.plotly_chart(styled(fig, fig.layout.height), width="stretch")
        with t3:
            st.dataframe(view, width="stretch", hide_index=True)

    st.subheader("Institutional Flows (FII / DII)")
    fl = q_flows()
    if fl.empty:
        no_data("Flows collection empty – run flows_pipeline.py.")
    else:
        if "data_source" in fl and (fl["data_source"] == "placeholder").any():
            st.caption("⚠️ Some flow rows come from the pipeline's placeholder fallback (NSE API unavailable).")
        recent = fl.tail(90)
        fig = go.Figure()
        fig.add_bar(x=recent["timestamp"], y=recent["fii_net"], name="FII net", marker_color=BLUE)
        fig.add_bar(x=recent["timestamp"], y=recent["dii_net"], name="DII net", marker_color=VIOLET)
        if "flow_zscore" in recent:
            fig.add_scatter(x=recent["timestamp"], y=recent["flow_zscore"], name="Flow z-score",
                            yaxis="y2", line=dict(color=AMBER))
        fig.update_layout(barmode="group", yaxis2=dict(overlaying="y", side="right", showgrid=False))
        st.plotly_chart(styled(fig, 380, "Net flows, ₹ Cr (last 90 sessions)"), width="stretch")

    st.subheader("Event Impact Analysis")
    asset = st.selectbox("Asset", KNOWN_ASSETS, key="ev_asset")
    feat = q_rows("features", asset, limit=1500)
    if feat.empty or "event_flag" not in feat:
        no_data("No feature rows with event flags for this asset.")
    else:
        feat["abs_return_%"] = feat["returns"].abs() * 100
        feat["day_type"] = np.where(feat["event_flag"].fillna(0) > 0, "Event day", "Non-event day")
        summary = feat.groupby("day_type").agg(sessions=("abs_return_%", "size"),
                                               mean_abs_return=("abs_return_%", "mean"),
                                               mean_volatility=("rolling_volatility", "mean")).round(4)
        c1, c2 = st.columns([1, 2])
        c1.dataframe(summary, width="stretch")
        fig = px.box(feat, x="day_type", y="abs_return_%", color="day_type",
                     color_discrete_map={"Event day": AMBER, "Non-event day": BLUE})
        fig.update_layout(showlegend=False)
        c2.plotly_chart(styled(fig, 340, f"{asset}: |daily return| on event vs non-event days"),
                        width="stretch")


def page_news():
    st.title("📰 News Sentiment Analysis")
    st.caption("Headlines scored with VADER (compound ≥ 0.05 positive, ≤ −0.05 negative).")
    c1, c2 = st.columns([1, 1])
    asset = c1.selectbox("Filter by asset keywords", ["All"] + KNOWN_ASSETS)
    limit = c2.slider("Articles", 50, 1000, 300, step=50)
    with st.spinner("Loading news…"):
        nw = q_news(None if asset == "All" else asset, limit=limit)
    if nw.empty:
        no_data("No matching news – run news_pipeline.py.")
    else:
        counts = nw["sentiment_label"].value_counts()
        m = st.columns(4)
        m[0].metric("Articles", len(nw))
        m[1].metric("Positive", int(counts.get("positive", 0)))
        m[2].metric("Negative", int(counts.get("negative", 0)))
        m[3].metric("Neutral", int(counts.get("neutral", 0)))

        g1, g2 = st.columns([1, 2])
        fig = px.pie(nw, names="sentiment_label", color="sentiment_label",
                     color_discrete_map=SENTIMENT_COLORS, hole=0.55)
        g1.plotly_chart(styled(fig, 340, "Sentiment mix"), width="stretch")
        daily = nw.set_index("timestamp")["sentiment_score"].resample("D").mean().dropna().reset_index()
        fig = px.bar(daily, x="timestamp", y="sentiment_score", color="sentiment_score",
                     color_continuous_scale=[RED, "#64748b", GREEN], range_color=[-0.5, 0.5])
        g2.plotly_chart(styled(fig, 340, "Daily average sentiment"), width="stretch")

        if "event_type" in nw:
            et = nw.groupby("event_type")["sentiment_score"].agg(["count", "mean"]).reset_index()
            fig = px.bar(et.sort_values("count"), y="event_type", x="count", color="mean", orientation="h",
                         color_continuous_scale=[RED, "#64748b", GREEN], range_color=[-0.5, 0.5])
            st.plotly_chart(styled(fig, 360, "Articles by tagged event type (colour = mean sentiment)"),
                            width="stretch")

        st.subheader("Latest headlines")
        show = nw[["timestamp", "headline", "source", "sentiment_label", "sentiment_score", "event_type", "link"]]
        st.dataframe(show, width="stretch", hide_index=True,
                     column_config={"link": st.column_config.LinkColumn("link", display_text="open")})

    st.subheader("Try VADER on a headline")
    text = st.text_input("Headline", "RBI keeps repo rate unchanged, signals support for growth")
    if text:
        try:
            score, label = get_vader()(text)
            st.markdown(f"**{label.upper()}** · compound score `{score:+.4f}`")
        except Exception as e:
            st.warning(f"VADER unavailable: {e}")


def page_social():
    st.title("📡 Social Anomaly Detection")
    st.caption("Isolation Forest on sentiment / buzz features + cross-domain stress flags (social_anomaly.py).")
    c1, c2 = st.columns([1, 1])
    asset = c1.selectbox("Asset", KNOWN_ASSETS, key="soc_asset")
    limit = c2.slider("Sessions", 30, 1000, 250, step=10)
    with st.spinner("Loading anomaly scores…"):
        soc = q_rows("social_anomaly", asset, limit=limit)
    if soc.empty:
        no_data("No social anomaly data – run social_anomaly.py.")
        return
    last = soc.iloc[-1]
    m = st.columns(4)
    m[0].metric("Anomaly score", fmt(last.get("anomaly_score"), "+.3f"))
    m[1].metric("Market stress", fmt(last.get("market_stress_score"), ".0f"))
    m[2].metric("Composite flag", "🚨 ANOMALY" if last.get("composite_anomaly_flag") == 1 else "Normal")
    m[3].metric(f"Anomalies in window", int(soc.get("composite_anomaly_flag", pd.Series(dtype=int)).fillna(0).sum()))

    fig = go.Figure()
    fig.add_scatter(x=soc["timestamp"], y=soc["anomaly_score"], name="Anomaly score", line=dict(color=CYAN))
    flagged = soc[soc.get("anomaly_flag", 0) == 1] if "anomaly_flag" in soc else soc.iloc[0:0]
    fig.add_scatter(x=flagged["timestamp"], y=flagged["anomaly_score"], mode="markers", name="Isolation Forest flag",
                    marker=dict(color=RED, size=9, symbol="x"))
    st.plotly_chart(styled(fig, 360, "Anomaly score history"), width="stretch")

    g1, g2 = st.columns(2)
    if "market_stress_score" in soc:
        fig = px.bar(soc, x="timestamp", y="market_stress_score", color_discrete_sequence=[AMBER])
        g1.plotly_chart(styled(fig, 300, "Market stress score"), width="stretch")
    if "social_sentiment_score" in soc:
        fig = px.line(soc, x="timestamp", y=["social_sentiment_score", "social_buzz_intensity"])
        g2.plotly_chart(styled(fig, 300, "Social sentiment & buzz"), width="stretch")
    with st.expander("Raw data"):
        st.dataframe(soc.iloc[::-1], width="stretch", hide_index=True)


def page_hedging():
    st.title("🛡️ Hedging Strategy")
    st.caption("Rule-based hedge-fund style recommendation from recent volatility, RSI, volume and sentiment (hedging.py).")
    asset = st.selectbox("Asset", KNOWN_ASSETS, key="hedge_asset")
    with st.spinner("Computing hedging metrics…"):
        h = q_hedging(asset)
    if not h or "error" in h:
        st.error(f"Hedging module error: {h.get('error') if h else 'no result'}")
        return
    m = st.columns(4)
    m[0].metric("Regime", h.get("regime", "—"))
    m[1].metric("Beta (proxy)", h.get("beta", "—"))
    m[2].metric("Volatility", h.get("volatility", "—"))
    m[3].metric("VaR 95%", h.get("var_95", "—"))
    m = st.columns(3)
    m[0].metric("Optimal hedge ratio", h.get("optimal_hedge_ratio", "—"))
    m[1].markdown(f"**Institutional flow**  \n{h.get('institutional_flow', '—')}")
    m[2].markdown(f"**Options skew**  \n{h.get('options_bias', '—')}")
    st.subheader(h.get("strategy_name", ""))
    st.write(h.get("strategy_desc", ""))
    if h.get("tools"):
        st.markdown("**Instruments:** " + " · ".join(f"`{t}`" for t in h["tools"]))


def page_model():
    st.title("🧠 Model Information")
    st.markdown("""
**Architecture – two-stage LightGBM (modeling.py)**
1. **Regime detection** – KMeans (k=3) on rolling volatility and return z-score.
2. **Stage 1 – impact classifier** (`LGBMClassifier`, multiclass): next-session direction −1 / 0 / +1
   (|forward return| < 0.2 % ⇒ neutral).
3. **SHAP feature selection** – drops the bottom 2 % of SHAP mass, then retrains.
4. **Stage 2 – magnitude regressor** (`LGBMRegressor`): expected absolute move.
5. **Backtest** – position = direction × magnitude, 0.1 % one-way transaction cost.

Inputs merge price technicals, news sentiment, FII/DII flows, macro-event surprises, social anomaly features,
cross-asset returns (Brent, INR/USD, Gold, Bitcoin) and lags.
""")
    try:
        clf, reg, selected, idx_to_label, enc = get_models()
    except Exception as e:
        st.error(f"Could not load model files from models/: {e}")
        return

    c1, c2 = st.columns([2, 1])
    imp = pd.DataFrame({"feature": selected, "importance": clf.feature_importances_}).sort_values("importance")
    fig = px.bar(imp, x="importance", y="feature", orientation="h", color_discrete_sequence=[BLUE])
    c1.plotly_chart(styled(fig, 820, f"Stage 1 split importance ({len(selected)} SHAP-selected features)"),
                    width="stretch")
    with c2:
        st.metric("Selected features", len(selected))
        st.metric("Stage 1 trees", getattr(clf, "n_estimators_", clf.n_estimators))
        st.metric("Stage 2 trees", getattr(reg, "n_estimators_", reg.n_estimators))
        st.write("**Class map:**", {int(k): int(v) for k, v in idx_to_label.items()})
        st.write("**Assets encoded:**", list(enc.classes_))
        with st.expander("Stage 1 hyper-parameters"):
            st.json({k: v for k, v in clf.get_params().items() if v is not None})
        with st.expander("Stage 2 hyper-parameters"):
            st.json({k: v for k, v in reg.get_params().items() if v is not None})

    st.subheader("Evaluation from stored predictions")
    st.caption("Computed from the `predictions` collection (backtest strategy returns and LightGBM/LSTM agreement). "
               "Training CV scores are logged by `modeling.py --train` and are not persisted.")
    rows = []
    with st.spinner("Aggregating backtest results…"):
        for a in KNOWN_ASSETS:
            p = q_rows("predictions", a, limit=2000)
            if p.empty:
                continue
            sr = p["strategy_return"].dropna() if "strategy_return" in p else pd.Series(dtype=float)
            agree = p["model_agreement"].dropna().astype(bool) if "model_agreement" in p else pd.Series(dtype=bool)
            rows.append({
                "asset": a, "predictions": len(p),
                "mean_confidence": p["confidence_score"].mean(),
                "backtest_total_return": sr.sum() if len(sr) else np.nan,
                "sharpe (ann.)": (sr.mean() / (sr.std() + 1e-9)) * np.sqrt(252) if len(sr) > 1 else np.nan,
                "win_rate": (sr > 0).mean() if len(sr) else np.nan,
                "lgbm_lstm_agreement": agree.mean() if len(agree) else np.nan,
            })
    if rows:
        st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
    else:
        no_data("No stored predictions to evaluate.")


def page_system():
    st.title("⚙️ System Status")
    client = mongo_or_none()
    c = st.columns(3)
    c[0].metric("MongoDB", "Connected" if client else "Unavailable")
    c[1].metric("Gemini API key", "Configured" if os.getenv("GEMINI_API_KEY") else "Missing")
    c[2].metric("Prometheus /metrics", "Enabled" if app_metrics.ENABLED else "Disabled")
    if client:
        st.dataframe(pd.DataFrame(q_counts().items(), columns=["collection", "documents"]),
                     width="stretch", hide_index=True)
    elif st.session_state.get("mongo_error"):
        st.error(st.session_state["mongo_error"])
    if st.button("Clear cache & reload"):
        st.cache_data.clear()
        st.rerun()


PAGES = {
    "Executive Dashboard": page_dashboard,
    "Market Prediction":   page_prediction,
    "Event Intelligence":  page_events,
    "News Sentiment":      page_news,
    "Social Anomaly":      page_social,
    "Hedging Strategy":    page_hedging,
    "Model Information":   page_model,
    "System Status":       page_system,
}


def main():
    with st.sidebar:
        st.markdown("## ⚡ EventOracle")
        st.caption("Project Apex · Event-Driven Indian Market Prediction")
        page = st.radio("Navigate", list(PAGES), label_visibility="collapsed")
        st.divider()
        if mongo_or_none() is None:
            st.error("MongoDB not connected. Set `MONGO_URI` in `.env` or Streamlit Secrets.")
        else:
            st.success("MongoDB Atlas connected")
        st.caption("Data cached for 10–30 min. For research/education only – not investment advice.")

    app_metrics.page_view(page)
    try:
        PAGES[page]()
    except Exception as e:
        st.error(f"Something went wrong rendering **{page}**: {e}")
        st.exception(e)


if __name__ == "__main__":
    main()
