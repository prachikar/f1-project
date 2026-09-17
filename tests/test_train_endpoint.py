"""
QA regression test for a crash in POST /train.

Bug: when `train_end_year` is set to the last season present in the data
(or later), the test split (years > train_end_year) is empty. That empty
DataFrame is handed to RandomForestClassifier.predict(), which raises
ValueError("Found array with 0 sample(s)... while a minimum of 1 is
required"). FastAPI has no handler for it, so the endpoint returns a bare
500 instead of a meaningful 4xx response.

Root cause: backend/app.py's `train` endpoint builds train/test masks
directly from `req.train_end_year` with no check that the test split is
non-empty (unlike `/forecast`, which does check for an empty *train*
split before calling train_model). See backend/app.py:578-582 and
src/evaluate.py:25.
"""

import httpx
import pytest

from backend.app import app
from src.load_data import load_f1_data

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _max_season() -> int:
    _constructors, constructor_standings, _races, _results = load_f1_data()
    return int(constructor_standings["season"].astype(int).max())


@pytest.fixture
async def client():
    # raise_app_exceptions=False mirrors a real deployment: uvicorn catches
    # unhandled exceptions and returns a 500 response rather than propagating
    # them into the caller, so the test observes what a real client would see.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


async def test_train_end_year_at_last_season_crashes_today(client):
    """
    Documents the current (buggy) behavior: train_end_year == the last
    season in the dataset leaves nothing to test on and crashes with a 500.

    Once backend/app.py validates that a non-empty test split exists
    (e.g. returning 422/400 with a clear message), this test should be
    updated to assert that response instead of a 500.
    """
    last_season = _max_season()

    response = await client.post("/train", json={"train_end_year": last_season})

    # This is the bug: an internal server error leaks out instead of a
    # clean validation error. Flip this assertion once the endpoint is fixed.
    assert response.status_code == 500


async def test_train_end_year_one_before_last_season_succeeds(client):
    """Sanity check: leaving at least one season for testing works fine."""
    last_season = _max_season()

    response = await client.post("/train", json={"train_end_year": last_season - 1})

    assert response.status_code == 200
    body = response.json()
    assert "accuracy" in body
    assert body["test_seasons"] >= 1


async def test_train_end_year_beyond_last_season_also_crashes_today(client):
    """A year past the end of the dataset hits the same empty-test-split bug."""
    last_season = _max_season()

    response = await client.post("/train", json={"train_end_year": last_season + 5})

    assert response.status_code == 500
