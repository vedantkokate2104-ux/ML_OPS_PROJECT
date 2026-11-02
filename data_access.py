"""
data_access.py
EventOracle – Shared MongoDB read layer

Framework-agnostic query helpers used by the Streamlit dashboard
(streamlit_app.py). Mirrors the queries served by api_server.py so both
frontends read identical data.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
from pymongo import MongoClient

DB_NAME = "eventoracle"
IST     = timezone(timedelta(hours=5, minutes=30))

KNOWN_ASSETS = [
    "NIFTY", "BANKNIFTY", "NIFTYIT", "NIFTYMETAL",
    "INRUSD", "BRENT", "GOLD", "SILVER", "PLATINUM", "BITCOIN",
]

# Asset → keywords for news filtering (same as api_server.py)
ASSET_KEYWORDS = {
    "NIFTY":      ["nifty", "sensex", "nse", "bse"],
    "BANKNIFTY":  ["bank nifty", "banknifty", "banking"],
    "NIFTYIT":    ["nifty it", "it sector", "tech stocks"],
    "NIFTYMETAL": ["nifty metal", "metal sector", "steel"],
    "INRUSD":     ["rupee", "inr", "dollar", "usd", "forex", "currency"],
    "BRENT":      ["oil", "brent", "crude", "opec", "petroleum"],
    "GOLD":       ["gold", "bullion", "yellow metal", "xauusd"],
    "SILVER":     ["silver", "xagusd"],
    "PLATINUM":   ["platinum", "pgm"],
    "BITCOIN":    ["bitcoin", "btc", "crypto", "cryptocurrency"],
}

NEWS_PROJECTION = {"_id": 0, "headline": 1, "source": 1, "timestamp": 1,
                   "sentiment_score": 1, "sentiment_label": 1, "event_type": 1, "link": 1}


# ─────────────────────────────────────────────
# CONNECTION
# ─────────────────────────────────────────────

def get_client(uri: Optional[str] = None) -> MongoClient:
    uri = uri or os.getenv("MONGO_URI")
    if not uri:
        raise ValueError("MONGO_URI not set")
    client = MongoClient(uri, serverSelectionTimeoutMS=10_000)
    client.admin.command("ping")
    return client


def _to_df(docs: list[dict]) -> pd.DataFrame:
    if not docs:
        return pd.DataFrame()
    df = pd.DataFrame(docs)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert(IST)
    return df


# ─────────────────────────────────────────────
# QUERIES
# ─────────────────────────────────────────────

def latest_prices(client: MongoClient) -> pd.DataFrame:
    """Latest close/return per tracked asset."""
    db, rows = client[DB_NAME], []
    for asset in KNOWN_ASSETS:
        doc = db["prices"].find_one({"asset": asset}, sort=[("timestamp", -1)], projection={"_id": 0}) or {}
        rows.append({"asset": asset, **doc})
    return _to_df(rows)


def prices(client: MongoClient, asset: str, days: Optional[int] = None, limit: int = 2000) -> pd.DataFrame:
    query: dict = {"asset": asset}
    if days:
        query["timestamp"] = {"$gte": datetime.now(timezone.utc) - timedelta(days=days)}
    docs = list(client[DB_NAME]["prices"].find(query, {"_id": 0}).sort("timestamp", -1).limit(limit))
    docs.reverse()
    return _to_df(docs)


def latest_rows(client: MongoClient, collection: str, asset: str, limit: int = 30) -> pd.DataFrame:
    """Most recent `limit` docs for an asset, returned in chronological order."""
    docs = list(client[DB_NAME][collection].find({"asset": asset}, {"_id": 0}).sort("timestamp", -1).limit(limit))
    docs.reverse()
    return _to_df(docs)


def news(client: MongoClient, asset: Optional[str] = None, limit: int = 100) -> pd.DataFrame:
    col = client[DB_NAME]["news"]
    query: dict = {}
    if asset:
        query = {"headline": {"$regex": "|".join(ASSET_KEYWORDS.get(asset, [asset.lower()])), "$options": "i"}}
    return _to_df(list(col.find(query, NEWS_PROJECTION).sort("timestamp", -1).limit(limit)))


def events(client: MongoClient, limit: int = 500) -> pd.DataFrame:
    return _to_df(list(client[DB_NAME]["events"].find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)))


def flows(client: MongoClient, limit: int = 365) -> pd.DataFrame:
    docs = list(client[DB_NAME]["flows"].find({}, {"_id": 0}).sort("timestamp", -1).limit(limit))
    docs.reverse()
    return _to_df(docs)


def collection_counts(client: MongoClient) -> dict:
    db = client[DB_NAME]
    return {n: db[n].estimated_document_count() for n in db.list_collection_names()}
