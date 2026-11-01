from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def kidney_model():
    """Classifier trained at test time on the labelled split of the kidney disease dataset

    Replaces a pickled model that could only be loaded with the scikit-learn version that produced it.
    The tests only need a fitted classifier with predict and predict_proba, not a good one.
    """
    train = pd.read_csv(DATA_DIR / "split" / "test_data.csv")
    model = RandomForestClassifier(n_estimators=50, random_state=0)
    model.fit(train.drop(columns=["class"]), train["class"])
    return model
