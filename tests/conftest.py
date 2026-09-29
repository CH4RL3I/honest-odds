import json
from pathlib import Path

import pytest

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def markets_fx():
    return json.loads((FIX / "markets.json").read_text())


@pytest.fixture(scope="session")
def history_fx():
    return json.loads((FIX / "history.json").read_text())


@pytest.fixture(scope="session")
def history_empty_fx():
    return json.loads((FIX / "history_empty.json").read_text())
