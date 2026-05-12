# src/pulse_analysis/pulse_finding_v2.py
"""Pulse finding shenangins for detector waveforms."""
import numpy as np

def _moving_average(x, window):
    window = int(window)
    x = np.asarray(x, dtype=float)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def _runs(mask):
    mask = np.asarray(mask, dtype=np.int8)
    edges = np.diff(np.r_[0, mask, 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts, ends))


def _find_pulse_groups_single(
    wf,
    baseline_rms_ch,
    hi_sigma=6.0,
    lo_sigma=3.0,
    smooth_window=5,
    max_gap=25,
    pad=10,
    min_width=5,
):
    """
    Find grouped pulses on a single-channel waveform. Returns list of
    dicts with `start` and `end` (end exclusive) and similar metadata.
    """
    wf = np.asarray(wf, dtype=float)
    n = wf.size

    smooth = _moving_average(wf, smooth_window)

    hi = hi_sigma * float(baseline_rms_ch)
    lo = lo_sigma * float(baseline_rms_ch)

    lo_runs = _runs(smooth >= lo)
    if not lo_runs:
        return []

    merged = []
    cur_s, cur_e = lo_runs[0]
    for s, e in lo_runs[1:]:
        gap = s - cur_e
        if gap <= max_gap:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    pulses = []
    for s, e in merged:
        if e - s < min_width:
            continue
        if not np.any(smooth[s:e] >= hi):
            continue

        start = max(0, s - pad)
        end = min(n, e + pad)

        region = wf[start:end]
        smooth_region = smooth[start:end]

        peak_rel = int(np.argmax(smooth_region))
        peak_idx = start + peak_rel

        pulse = {
            "start": start,
            "end": end,
            "peak_index": peak_idx,
            "peak_amplitude": float(wf[peak_idx]),
            "sum": float(np.sum(region)),
            "width": int(end - start),
            "n_local_max": _count_local_maxima(smooth_region),
        }
        pulses.append(pulse)

    return pulses


def _count_local_maxima(y):
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return 0
    return int(np.sum((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])))


def find_pulse_groups(
    event,
    hi_sigma=6.0,
    lo_sigma=3.0,
    smooth_window=5,
    max_gap=25,
    pad=10,
    min_width=5,
):
    """
    Function looks at a single trace
    """

    # Summed waveform across the two PMTs
    wfs = np.asarray(event["wfs"], dtype=float)
    summed = wfs.sum(axis=0)

    # Get noise baseline level from channel sum
    baseline_rms = np.asarray(event["baseline_rms"], dtype=float)
    noise_sigma = float(np.sqrt(np.sum(baseline_rms ** 2)))

    # Create smoothed version of pulse
    smooth = _moving_average(summed, smooth_window)

    # Define high and low thresholds
    hi = hi_sigma * noise_sigma
    lo = lo_sigma * noise_sigma

    # Find where the smoothed summed pulse goes over the low threshold
    lo_runs = _runs(smooth >= lo)
    if not lo_runs:  # Return empty pulse
        return [], summed, smooth

    # Merge regions that are close together (within max_gap)
    merged = []
    cur_s, cur_e = lo_runs[0]
    for s, e in lo_runs[1:]:
        gap = s - cur_e
        if gap <= max_gap:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    pulses = []
    n = len(summed)

    # Do the coincidence check by pulse finding on single channels
    ch_pulses = []
    for ch in range(wfs.shape[0]):
        ch_p = _find_pulse_groups_single(
            wfs[ch, :], baseline_rms[ch],
            hi_sigma=hi_sigma, lo_sigma=lo_sigma,
            smooth_window=smooth_window, max_gap=max_gap,
            pad=pad, min_width=min_width,
        )
        ch_pulses.append(ch_p)

    # Loop over merged because these are now our physical regions 
    for s, e in merged:
        if e - s < min_width:
            continue

        # Require at least one sample above the high threshold in the group
        if not np.any(smooth[s:e] >= hi):
            continue

        # Cut the start and end
        start = max(0, s - pad)
        end = min(n, e + pad)

        # Get the cut regions of the found pulse
        region = summed[start:end]
        smooth_region = smooth[start:end]

        peak_rel = int(np.argmax(smooth_region))
        peak_idx = start + peak_rel

        pulse = {
            "start": start,
            "end": end,
            "peak_index": peak_idx,
            "peak_amplitude": float(summed[peak_idx]),
            "sum": float(np.sum(region)),
            "width": int(end - start),
            "two_channel_pulse": bool(
                any((p["start"] < end and p["end"] > start) for p in ch_pulses[0])
                and any((p["start"] < end and p["end"] > start) for p in ch_pulses[1])
            ),
        }
        pulses.append(pulse)

    return pulses, summed, smooth


def find_all_pulses(data, **kwargs):
    """
    Loop thorugh lots of pulses at once, envoking the single-event function
    """
    results = []

    for event in data:
        pulses, summed, smooth = find_pulse_groups(event, **kwargs)
        results.append({
            "event_index": int(event["event_index"]),
            "event_time": float(event["event_time"]),
            "pulses": pulses,
        })

    return results