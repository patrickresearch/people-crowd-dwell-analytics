import argparse
import json
import time
from functools import partial
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from crowd_dwell.zones import Scene, draw_scene, load_scene, read_frames, resolve_source

COLUMNS = ["frame", "t", "track_id", "x1", "y1", "x2", "y2", "conf", "in_zone", "crossed"]
ANCHOR = (sv.Position.BOTTOM_CENTER,)
BG_SAMPLES = 25


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Detect and track people, write tracks + video.")
    ap.add_argument("source", help="video file or MOT sequence directory")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--zones", default="configs/zones.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--tracker", default="bytetrack.yaml")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max-frames", type=int)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--no-anonymize", action="store_true")
    return ap.parse_args()


def tracked(result) -> sv.Detections:
    det = sv.Detections.from_ultralytics(result)
    if det.tracker_id is None:
        det = det[np.zeros(len(det), dtype=bool)]
        det.tracker_id = np.empty(0, dtype=int)
    return det


class Renderer:
    def __init__(self, scene: Scene, fps: float, anonymize: bool):
        self.scene = scene
        self.anon = sv.PixelateAnnotator(pixel_size=16) if anonymize else None
        lookup = sv.ColorLookup.TRACK
        self.trace = sv.TraceAnnotator(
            position=sv.Position.BOTTOM_CENTER, trace_length=int(2 * fps), color_lookup=lookup
        )
        self.box = sv.BoxAnnotator(thickness=2, color_lookup=lookup)
        self.label = sv.LabelAnnotator(text_scale=0.5, text_padding=4, color_lookup=lookup)

    def __call__(self, frame, det, labels, counts) -> np.ndarray:
        if self.anon:
            frame = self.anon.annotate(frame.copy(), det)
        frame = draw_scene(frame, self.scene)
        frame = self.trace.annotate(frame, det)
        frame = self.box.annotate(frame, det)
        frame = self.label.annotate(frame, det, labels=labels)
        text = "in {}   out {}   zone {}".format(*counts)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
        cv2.rectangle(frame, (20, 20), (44 + tw, 44 + th + 12), (0, 0, 0), -1)
        cv2.putText(
            frame, text, (32, 32 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2,
            cv2.LINE_AA,
        )
        return frame


def background(frames: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(frames)
    parts = [np.median(s, axis=0) for s in np.array_split(stack, 8, axis=1)]
    return np.concatenate(parts).astype(np.uint8)


def main() -> None:
    a = parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = load_scene(a.zones, a.scene)
    source, fps = resolve_source(a.source)

    line = sv.LineZone(
        start=sv.Point(*scene.line[0]), end=sv.Point(*scene.line[1]), triggering_anchors=ANCHOR
    )
    zone = sv.PolygonZone(polygon=scene.zone, triggering_anchors=ANCHOR)
    render = None if a.no_video else Renderer(scene, fps, not a.no_anonymize)
    writer = None

    model = YOLO(a.model)
    device = a.device if a.model.endswith(".pt") else None
    track = partial(
        model.track, persist=True, tracker=a.tracker, classes=[0], conf=a.conf,
        imgsz=a.imgsz, device=device, verbose=False,
    )

    rows, samples, dwell = [], [], {}
    bg_step = max(1, int(fps))
    t0 = time.perf_counter()
    n = 0
    for n, frame in enumerate(read_frames(source), start=1):
        det = tracked(track(frame)[0])
        crossed_in, crossed_out = line.trigger(det)
        in_zone = zone.trigger(det)
        crossed = crossed_in.astype(int) - crossed_out.astype(int)
        for tid in det.tracker_id[in_zone]:
            dwell[tid] = dwell.get(tid, 0) + 1
        rows.append(
            np.column_stack([
                np.full(len(det), n), np.full(len(det), (n - 1) / fps), det.tracker_id,
                det.xyxy, det.confidence, in_zone, crossed,
            ])
        )
        if (n - 1) % bg_step == 0 and len(samples) < BG_SAMPLES:
            samples.append(frame)
        if render:
            labels = [
                f"#{t} {dwell[t] / fps:.1f}s" if z else f"#{t}"
                for t, z in zip(det.tracker_id, in_zone, strict=True)
            ]
            vis = render(frame, det, labels, (line.in_count, line.out_count, in_zone.sum()))
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(
                    str(out / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            writer.write(vis)
        if a.max_frames and n >= a.max_frames:
            break
    elapsed = time.perf_counter() - t0
    if writer:
        writer.release()

    df = pd.DataFrame(np.concatenate(rows), columns=COLUMNS)
    df = df.astype({c: int for c in ("frame", "track_id", "in_zone", "crossed")})
    df.round(2).to_csv(out / "tracks.csv", index=False)
    cv2.imwrite(str(out / "background.jpg"), background(samples))
    h, w = samples[0].shape[:2]
    meta = {
        "source": a.source, "scene": a.scene, "model": a.model, "tracker": a.tracker,
        "imgsz": a.imgsz, "device": device, "fps": fps, "frames": n, "width": w, "height": h,
        "throughput_fps": round(n / elapsed, 2), "line_in": line.in_count,
        "line_out": line.out_count, "tracks": int(df.track_id.nunique()),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
