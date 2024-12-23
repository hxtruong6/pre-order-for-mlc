import pytest
import numpy as np
from base_classifiers import BaseClassifiers
from estimator import Estimator


@pytest.fixture
def sample_data():
    X = np.random.rand(100, 10)
    Y = np.random.randint(0, 2, (100, 5))
    return X, Y


def test_pairwise_calibrated_classifier(sample_data):
    X, Y = sample_data
    base_clf = BaseClassifiers(Estimator("RF"))
    pairwise_clf, calibrated_clf = base_clf.pairwise_calibrated_classifier(X, Y)

    assert len(pairwise_clf) == (Y.shape[1] * (Y.shape[1] - 1)) // 2
