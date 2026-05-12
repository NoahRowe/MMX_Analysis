# src/pulse_analysis/pulse_finding_v1.py
"""Pulse finding shenangins for detector waveforms."""

from typing import Optional

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm


HitDType = np.dtype(
    [
        ("event_index", np.int32),
        ("event_time", np.float64),
        ("channel", np.int16),
        ("hit_area_raw", np.float64),
        ("hit_start", np.int16),
        ("hit_end", np.int16),
        ("hit_left_bound", np.int16),
        ("hit_right_bound", np.int16),
    ]
)


def find_hits_in_waveform(
    wf: NDArray[np.floating],
    threshold: float,
    event_index: int,
    event_time: float,
    channel: int,
    left_ext: int = 10,
    right_ext: int = 20,
) -> list[tuple[int, float, int, float, int, int, int, int]]:

    hits: list[tuple[int, float, int, float, int, int, int, int]] = []

    above_count = 0
    below_count = 0
    in_hit = False
    hit_start: Optional[int] = None

    for j, sample in enumerate(wf):
        if sample > threshold:
            above_count += 1
            below_count = 0

            if above_count < 2:
                continue

            if not in_hit:
                in_hit = True
                hit_start = j - 1

        else:
            above_count = 0

            if not in_hit:
                continue

            below_count += 1
            if below_count < 2:
                continue

            assert hit_start is not None
            hit_end = j - 1
            hit_left_bound = hit_start - left_ext
            hit_right_bound = hit_end + right_ext
            hit_area_raw = float(np.sum(wf[hit_left_bound:hit_right_bound]))

            hits.append(
                (
                    event_index,
                    event_time,
                    channel,
                    hit_area_raw,
                    hit_start,
                    hit_end,
                    hit_left_bound,
                    hit_right_bound,
                )
            )

            in_hit = False
            above_count = 0
            below_count = 0
            hit_start = None

    return hits


def find_hits(
    data: np.ndarray,
    n_sigma_hitfinder_threshold: float = 4.0,
    hit_left_extension: int = 10,
    hit_right_extension: int = 20,
    show_progress: bool = True,
) -> np.ndarray:

    all_hits: list[tuple[int, float, int, float, int, int, int, int]] = []

    n_channels = data.dtype["wfs"].shape[0]
    event_iter = tqdm(data) if show_progress else data

    for d in event_iter:
        for ch in range(n_channels):
            wf = d["wfs"][ch]
            threshold = d["baseline_rms"][ch] * n_sigma_hitfinder_threshold

            all_hits.extend(
                find_hits_in_waveform(
                    wf=wf,
                    threshold=threshold,
                    event_index=int(d["event_index"]),
                    event_time=float(d["event_time"]),
                    channel=ch,
                    left_ext=hit_left_extension,
                    right_ext=hit_right_extension,
                )
            )

    return np.array(all_hits, dtype=HitDType)
