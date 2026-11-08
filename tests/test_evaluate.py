import numpy as np
import pandas as pd
import pytest

from crowd_dwell.evaluate import evaluate, iou


def make_gt(n_frames=10):
    rows = []
    for f in range(1, n_frames + 1):
        rows.append((f, 1, 100 + f, 100, 50, 120, 1, 1, 1.0))
        rows.append((f, 2, 400 - f, 300, 50, 120, 1, 1, 1.0))
    return pd.DataFrame(rows, columns=["frame", "id", "x", "y", "w", "h", "flag", "cls", "vis"])


def as_pred(gt):
    return gt[["frame", "id", "x", "y", "w", "h"]].copy()


def test_iou_identity_and_disjoint():
    a = np.array([[0, 0, 10, 10], [100, 100, 10, 10]], float)
    s = iou(a, a)
    assert s[0, 0] == pytest.approx(1)
    assert s[0, 1] == 0


def test_perfect_tracking():
    gt = make_gt()
    m = evaluate(gt, as_pred(gt))
    assert m["MOTA"] == pytest.approx(1)
    assert m["IDF1"] == pytest.approx(1)
    assert m["IDSW"] == 0


def test_id_switch_counts_once_and_lowers_idf1():
    gt = make_gt()
    pred = as_pred(gt)
    pred.loc[(pred.id == 1) & (pred.frame > 5), "id"] = 9
    m = evaluate(gt, pred)
    assert m["IDSW"] == 1
    assert m["IDF1"] == pytest.approx(0.75)
    assert m["MOTA"] == pytest.approx(1 - 1 / 20)


def test_prediction_on_distractor_is_ignored():
    gt = make_gt()
    distractor = pd.DataFrame(
        [(f, 3, 800, 300, 50, 120, 0, 7, 1.0) for f in range(1, 11)], columns=gt.columns
    )
    pred = pd.concat([as_pred(gt), as_pred(distractor)])
    m = evaluate(pd.concat([gt, distractor]), pred)
    assert m["FP"] == 0
    assert m["MOTA"] == pytest.approx(1)
