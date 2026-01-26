import numpy as np

from src.eval.eval_classification import wcr


def test_wcr():
    mat = np.array([[0.0, 0.2], [0.3, 0.0]])
    assert wcr(mat) == 0.3
