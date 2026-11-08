import argparse
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

IOU_THR = 0.5
PEDESTRIAN = 1
DISTRACTORS = (2, 7, 8, 12)
GT_COLUMNS = ["frame", "id", "x", "y", "w", "h", "flag", "cls", "vis"]

Frame = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


def load_gt(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=GT_COLUMNS)


def load_pred(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return pd.DataFrame({
        "frame": df.frame, "id": df.track_id, "x": df.x1, "y": df.y1,
        "w": df.x2 - df.x1, "h": df.y2 - df.y1,
    })


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a1, a2 = a[:, None, :2], a[:, None, :2] + a[:, None, 2:]
    b1, b2 = b[None, :, :2], b[None, :, :2] + b[None, :, 2:]
    inter = np.clip(np.minimum(a2, b2) - np.maximum(a1, b1), 0, None).prod(-1)
    union = a[:, None, 2:].prod(-1) + b[None, :, 2:].prod(-1) - inter
    return inter / np.maximum(union, 1e-9)


def assign(score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r, c = linear_sum_assignment(score, maximize=True)
    keep = score[r, c] > 0
    return r[keep], c[keep]


def frames(gt: pd.DataFrame, pred: pd.DataFrame) -> Iterator[Frame]:
    box = ["x", "y", "w", "h"]
    gt_f, pr_f = dict(tuple(gt.groupby("frame"))), dict(tuple(pred.groupby("frame")))
    for f in sorted(gt_f.keys() | pr_f.keys()):
        g, p = gt_f.get(f, gt.iloc[:0]), pr_f.get(f, pred.iloc[:0])
        pb = p[box].to_numpy(float)
        s = iou(g[box].to_numpy(float), pb)
        r, c = assign(np.where(s >= IOU_THR, s, 0))
        keep = np.ones(len(p), bool)
        keep[c[np.isin(g.cls.to_numpy()[r], DISTRACTORS)]] = False
        g = g[(g.flag == 1) & (g.cls == PEDESTRIAN)]
        yield g.id.to_numpy(), g[box].to_numpy(float), p.id.to_numpy()[keep], pb[keep]


def evaluate(gt: pd.DataFrame, pred: pd.DataFrame) -> dict[str, float]:
    tp = idsw = n_gt = n_pr = 0
    iou_sum = 0.0
    last, prev, pairs = {}, {}, Counter()
    for g_id, g_box, p_id, p_box in frames(gt, pred):
        n_gt += len(g_id)
        n_pr += len(p_id)
        s = iou(g_box, p_box)
        valid = s >= IOU_THR
        vi, vj = np.nonzero(valid)
        pairs.update(zip(g_id[vi], p_id[vj], strict=True))
        cont = np.array([[prev.get(a) == b for b in p_id] for a in g_id], bool).reshape(s.shape)
        r, c = assign(np.where(valid, s + 1000 * cont, 0))
        cur = dict(zip(g_id[r], p_id[c], strict=True))
        idsw += sum(a in last and last[a] != b for a, b in cur.items())
        last.update(cur)
        prev = cur
        tp += len(r)
        iou_sum += s[r, c].sum()

    gu, gi = np.unique([k[0] for k in pairs], return_inverse=True)
    pu, pi = np.unique([k[1] for k in pairs], return_inverse=True)
    m = np.zeros((len(gu), len(pu)))
    m[gi, pi] = list(pairs.values())
    idtp = m[linear_sum_assignment(m, maximize=True)].sum()

    fn, fp = n_gt - tp, n_pr - tp
    return {
        "MOTA": 1 - (fn + fp + idsw) / n_gt,
        "IDF1": 2 * idtp / (n_gt + n_pr),
        "MOTP": iou_sum / max(tp, 1),
        "Recall": tp / n_gt,
        "Precision": tp / max(n_pr, 1),
        "IDSW": idsw,
        "FP": fp,
        "FN": fn,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="MOTA / IDF1 against MOTChallenge ground truth.")
    ap.add_argument("gt", help="gt/gt.txt of a MOT17 sequence")
    ap.add_argument("tracks", nargs="+", help="tracks.csv files to compare")
    a = ap.parse_args()
    gt = load_gt(a.gt)
    rows = {Path(t).parent.name + "/" + Path(t).stem: evaluate(gt, load_pred(t)) for t in a.tracks}
    table = pd.DataFrame(rows).T
    pct = ["MOTA", "IDF1", "MOTP", "Recall", "Precision"]
    table[pct] = (table[pct] * 100).round(1)
    table = table.astype({c: int for c in ("IDSW", "FP", "FN")})
    print(table.to_string())


if __name__ == "__main__":
    main()
