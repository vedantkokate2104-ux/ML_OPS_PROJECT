"""
Smoke tests for the Streamlit dashboard.

Every page must render without an uncaught exception – with a live
MONGO_URI (local) or without one (CI), where pages degrade gracefully.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")
PAGES = [
    "Executive Dashboard", "Market Prediction", "Event Intelligence", "News Sentiment",
    "Social Anomaly", "Hedging Strategy", "Model Information", "System Status",
]


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(APP, default_timeout=240).run()


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception, [e.value for e in app.exception]
