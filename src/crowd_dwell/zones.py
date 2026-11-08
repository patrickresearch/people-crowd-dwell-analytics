import argparse
import configparser
import queue
import threading
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

LINE_COLOR = (255, 255, 255)
ZONE_COLOR = (214, 120, 42)


@dataclass(frozen=True)
class Scene:
    line: np.ndarray
    zone: np.ndarray


def load_scene(path: str | Path, name: str) -> Scene:
    cfg = yaml.safe_load(Path(path).read_text())[name]
    return Scene(np.array(cfg["line"], dtype=int), np.array(cfg["zone"], dtype=np.int32))


def resolve_source(source: str | Path) -> tuple[str, float]:
    p = Path(source)
    ini = p / "seqinfo.ini"
    if ini.exists():
        cfg = configparser.ConfigParser()
        cfg.read(ini)
        seq = cfg["Sequence"]
        return str(p / seq["imDir"]), float(seq["frameRate"])
    cap = cv2.VideoCapture(str(p))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return str(p), fps


def _images(files: list[Path], workers: int, prefetch: int) -> Iterator[np.ndarray]:
    with ThreadPoolExecutor(workers) as ex:
        pending = deque()
        for f in files:
            pending.append(ex.submit(cv2.imread, str(f)))
            if len(pending) >= prefetch:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def _video(path: str, prefetch: int) -> Iterator[np.ndarray]:
    q = queue.Queue(prefetch)

    def run() -> None:
        cap = cv2.VideoCapture(path)
        while (frame := cap.read()[1]) is not None:
            q.put(frame)
        q.put(None)
        cap.release()

    threading.Thread(target=run, daemon=True).start()
    while (frame := q.get()) is not None:
        yield frame


def read_frames(source: str, workers: int = 4, prefetch: int = 16) -> Iterator[np.ndarray]:
    p = Path(source)
    if p.is_dir():
        return _images(sorted(p.glob("*.jpg")), workers, prefetch)
    return _video(str(p), prefetch)


def first_frame(source: str) -> np.ndarray:
    return next(iter(read_frames(source, workers=1, prefetch=1)))


def draw_scene(frame: np.ndarray, scene: Scene) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()
    cv2.fillPoly(overlay, [scene.zone], ZONE_COLOR)
    cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
    cv2.polylines(out, [scene.zone], True, ZONE_COLOR, 3, cv2.LINE_AA)
    cv2.line(out, tuple(scene.line[0]), tuple(scene.line[1]), LINE_COLOR, 3, cv2.LINE_AA)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Render line and zone on the first frame.")
    ap.add_argument("source")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--zones", default="configs/zones.yaml")
    ap.add_argument("--out", default="runs/preview.jpg")
    a = ap.parse_args()
    src, _ = resolve_source(a.source)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(a.out, draw_scene(first_frame(src), load_scene(a.zones, a.scene)))
    print(a.out)


if __name__ == "__main__":
    main()
