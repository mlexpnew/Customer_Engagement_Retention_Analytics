from pathlib import Path

import pandas as pd
import pytest

from churnshield.data import add_features, load, split
from churnshield.model import train

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def data() -> pd.DataFrame:
    return add_features(load(ROOT / "data" / "European_Bank.xlsx"))


@pytest.fixture(scope="session")
def small_bundle(data):
    """A quick model on 2,000 rows, so tests do not depend on the committed artifact."""
    from churnshield.explain import reference_profile

    tr, _ = split(data.sample(2000, random_state=1))
    bundle, _ = train(tr, folds=3)
    bundle.metadata["reference_profile"] = reference_profile(tr)
    bundle.reference = tr
    return bundle


@pytest.fixture
def customers(data) -> pd.DataFrame:
    return data.drop(columns=["Exited"]).sample(200, random_state=5)
