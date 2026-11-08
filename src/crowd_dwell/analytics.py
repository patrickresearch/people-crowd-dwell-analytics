import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from crowd_dwell.zones import draw_scene, load_scene

SERIES = ("#2a78d6", "#eb6834")
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"


def load_tracks(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.assign(fx=(df.x1 + df.x2) / 2, fy=df.y2)


def stitch(df: pd.DataFrame, fps: float, max_gap_s: float, max_dist_px: float) -> pd.DataFrame:
    g = df.sort_values("frame").groupby("track_id")[["frame", "fx", "fy"]]
    head, tail = g.first().sort_values("frame"), g.last()
    max_gap = max_gap_s * fps
    root, open_tails = {}, set()
    for tid, h in head.iterrows():
        best, best_d = None, max_dist_px
        for a in open_tails:
            gap = h.frame - tail.at[a, "frame"]
            d = np.hypot(h.fx - tail.at[a, "fx"], h.fy - tail.at[a, "fy"])
            if 0 < gap <= max_gap and d <= best_d:
                best, best_d = a, d
        if best is not None:
            open_tails.discard(best)
            root[tid] = root.get(best, best)
        open_tails.add(tid)
    return df.assign(track_id=df.track_id.map(root).fillna(df.track_id).astype(int))


def track_kpis(df: pd.DataFrame, fps: float) -> pd.DataFrame:
    d = df.sort_values(["track_id", "frame"])
    step = np.hypot(d.groupby("track_id").fx.diff(), d.groupby("track_id").fy.diff())
    g = d.assign(step=step, cin=d.crossed == 1, cout=d.crossed == -1).groupby("track_id")
    return pd.DataFrame({
        "first_s": g.t.min(),
        "last_s": g.t.max(),
        "dwell_s": g.in_zone.sum() / fps,
        "path_px": g.step.sum(),
        "crossed_in": g.cin.sum(),
        "crossed_out": g.cout.sum(),
    }).round(2)


def throughput(df: pd.DataFrame, bin_s: float) -> pd.DataFrame:
    b = (df.t // bin_s * bin_s).rename("t_s")
    g = df.assign(cin=df.crossed == 1, cout=df.crossed == -1).groupby(b)
    scale = 60 / bin_s
    return pd.DataFrame({
        "persons": g.track_id.nunique(),
        "in_per_min": g.cin.sum() * scale,
        "out_per_min": g.cout.sum() * scale,
    })


def heatmap(df: pd.DataFrame, bg: np.ndarray, sigma: float) -> np.ndarray:
    h, w = bg.shape[:2]
    acc = np.zeros((h, w), np.float32)
    ys = df.fy.to_numpy().astype(int).clip(0, h - 1)
    xs = df.fx.to_numpy().astype(int).clip(0, w - 1)
    np.add.at(acc, (ys, xs), 1)
    acc = np.log1p(cv2.GaussianBlur(acc, (0, 0), sigma) * sigma**2)
    norm = (np.clip(acc / max(np.percentile(acc, 99.5), 1e-9), 0, 1) * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
    alpha = (norm.astype(np.float32) / 255)[..., None] * 0.75
    return (bg * (1 - alpha) + color * alpha).astype(np.uint8)


def figure(title: str, x_title: str, y_title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(color=INK, size=18)),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(color=MUTED, size=13),
        width=900, height=420, margin=dict(l=60, r=30, t=60, b=50), bargap=0.08,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"),
        hovermode="x unified",
    )
    fig.update_xaxes(title=x_title, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(title=y_title, gridcolor=GRID, zeroline=False, rangemode="tozero")
    return fig


def save(fig: go.Figure, out: Path, name: str) -> None:
    fig.write_html(out / f"{name}.html", include_plotlyjs="cdn")
    fig.write_image(out / f"{name}.png", scale=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate tracks into KPIs, plots and heatmap.")
    ap.add_argument("run", help="output directory of crowd-track")
    ap.add_argument("--zones", default="configs/zones.yaml")
    ap.add_argument("--max-gap", type=float, default=1.0, help="seconds")
    ap.add_argument("--max-dist", type=float, default=80.0, help="pixels")
    ap.add_argument("--min-dwell", type=float, default=0.5, help="seconds")
    ap.add_argument("--bin", type=float, default=5.0, help="seconds")
    ap.add_argument("--sigma", type=float, default=15.0)
    a = ap.parse_args()

    run = Path(a.run)
    meta = json.loads((run / "meta.json").read_text())
    fps = meta["fps"]
    raw = load_tracks(run / "tracks.csv")
    df = stitch(raw, fps, a.max_gap, a.max_dist)
    df.drop(columns=["fx", "fy"]).to_csv(run / "tracks_stitched.csv", index=False)

    kpis = track_kpis(df, fps)
    kpis.to_csv(run / "track_kpis.csv")
    tp = throughput(df, a.bin)
    tp.to_csv(run / "throughput.csv")
    dwell = kpis.dwell_s[kpis.dwell_s >= a.min_dwell]

    bg = cv2.imread(str(run / "background.jpg"))
    scene = load_scene(a.zones, meta["scene"])
    cv2.imwrite(
        str(run / "heatmap.jpg"), draw_scene(heatmap(df, bg, a.sigma), scene),
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )

    fig = figure("Dwell time in zone", "seconds", "persons")
    fig.add_histogram(x=dwell, marker_color=SERIES[0], name="dwell", xbins=dict(size=1))
    peak = np.bincount(dwell.astype(int)).max() if dwell.size else 1
    fig.update_layout(showlegend=False, hovermode="closest")
    fig.update_yaxes(dtick=max(1, int(np.ceil(peak / 5))))
    save(fig, run, "dwell_hist")

    fig = figure(f"Line crossings per minute ({a.bin:g}s bins)", "time [s]", "persons / min")
    for col, name, c in zip(("in_per_min", "out_per_min"), ("in", "out"), SERIES, strict=True):
        fig.add_scatter(
            x=tp.index, y=tp[col], name=name, mode="lines",
            line=dict(color=c, width=2, shape="hv"),
        )
    save(fig, run, "throughput")

    summary = {
        "tracks_raw": int(raw.track_id.nunique()),
        "tracks_stitched": int(df.track_id.nunique()),
        "line_in": int(kpis.crossed_in.sum()),
        "line_out": int(kpis.crossed_out.sum()),
        "zone_visitors": int(dwell.size),
        "dwell_median_s": round(float(dwell.median()), 2) if dwell.size else None,
        "dwell_p90_s": round(float(dwell.quantile(0.9)), 2) if dwell.size else None,
        "persons_per_min": round(df.track_id.nunique() / (meta["frames"] / fps) * 60, 1),
    }
    (run / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
