import pandas as pd

from crowd_dwell.analytics import stitch, throughput, track_kpis

FPS = 10.0


def track(tid, frames, x0, in_zone=0, crossed_at=None):
    return [
        {
            "frame": f, "t": (f - 1) / FPS, "track_id": tid,
            "fx": x0 + f, "fy": 500.0, "in_zone": in_zone,
            "crossed": int(f == crossed_at),
        }
        for f in frames
    ]


def test_stitch_merges_short_gap_and_keeps_far_tracks():
    df = pd.DataFrame(
        track(1, range(1, 11), 100) + track(2, range(15, 25), 104) + track(3, range(15, 25), 900)
    )
    out = stitch(df, FPS, max_gap_s=1.0, max_dist_px=50)
    assert sorted(out.track_id.unique()) == [1, 3]


def test_stitch_respects_max_gap():
    df = pd.DataFrame(track(1, range(1, 11), 100) + track(2, range(30, 40), 120))
    assert stitch(df, FPS, max_gap_s=1.0, max_dist_px=500).track_id.nunique() == 2


def test_stitch_chains_transitively():
    df = pd.DataFrame(
        track(1, range(1, 6), 0) + track(2, range(8, 13), 0) + track(3, range(15, 20), 0)
    )
    assert stitch(df, FPS, max_gap_s=1.0, max_dist_px=50).track_id.unique().tolist() == [1]


def test_kpis_dwell_and_crossings():
    df = pd.DataFrame(track(1, range(1, 21), 0, in_zone=1, crossed_at=5))
    k = track_kpis(df, FPS).loc[1]
    assert k.dwell_s == 2.0
    assert k.crossed_in == 1
    assert k.path_px == 19.0


def test_throughput_scales_to_minutes():
    df = pd.DataFrame(track(1, range(1, 51), 0, crossed_at=3))
    tp = throughput(df, bin_s=5.0)
    assert tp.in_per_min.iloc[0] == 12
    assert tp.persons.iloc[0] == 1
