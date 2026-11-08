import argparse
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

from crowd_dwell.zones import first_frame, read_frames, resolve_source


def load(weights: str, backend: str, shape: tuple[int, int]) -> tuple[YOLO, str | None]:
    if backend.startswith("coreml"):
        half = backend == "coreml-fp16"
        return YOLO(YOLO(weights).export(format="coreml", imgsz=list(shape), half=half)), None
    return YOLO(weights), backend


def measure(model, device, source, imgsz, frames, warmup) -> dict[str, float]:
    track = partial(
        model.track, persist=True, classes=[0], conf=0.1, imgsz=imgsz, device=device,
        verbose=False,
    )
    dt, speed = [], []
    t = time.perf_counter()
    for i, frame in enumerate(read_frames(source)):
        r = track(frame)[0]
        now = time.perf_counter()
        if i >= warmup:
            dt.append(now - t)
            speed.append(r.speed)
        t = now
        if i + 1 >= warmup + frames:
            break
    ms = pd.DataFrame(speed).median().round(1)
    return {
        "fps": round(len(dt) / sum(dt), 1),
        "pre_ms": float(ms.preprocess),
        "infer_ms": float(ms.inference),
        "post_ms": float(ms.postprocess),
        "frame_ms": round(float(np.median(dt)) * 1000, 1),
        "p95_frame_ms": round(float(np.percentile(dt, 95)) * 1000, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="End-to-end detect+track FPS per model and backend.")
    ap.add_argument("source")
    ap.add_argument("--models", nargs="+", default=["yolo26n.pt", "yolo26s.pt", "yolo26m.pt"])
    ap.add_argument("--backends", nargs="+", default=["mps", "coreml-fp32", "coreml-fp16"])
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--out", default="runs/bench.csv")
    a = ap.parse_args()

    source, _ = resolve_source(a.source)
    h, w = first_frame(source).shape[:2]
    shape = (math.ceil(h * a.imgsz / w / 32) * 32, a.imgsz)
    rows = []
    for weights in a.models:
        for backend in a.backends:
            model, device = load(weights, backend, shape)
            res = measure(model, device, source, a.imgsz, a.frames, a.warmup)
            rows.append({"model": Path(weights).stem, "backend": backend, **res})
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(a.out, index=False)
    print(f"\ninput {shape[1]}x{shape[0]}, {a.frames} frames after {a.warmup} warmup\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
