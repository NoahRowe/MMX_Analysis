# src/pulse_analysis/io.py
import numpy as np
import uproot


def read_root_file(filename, max_events=None):
    with uproot.open(filename) as f:
        print("Raw file keys:", f.keys())
        tree = f["t1"]

        if max_events is None:
            data = tree.arrays(library="numpy")
        else:
            data = tree.arrays(entry_stop=max_events, library="numpy")

        return data


def _window_rms(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.std(x))


def _window_mean(x):
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x))


def _estimate_baseline_strict(wf, n_baseline_samples=50, rms_threshold=0.01):

    wf = np.asarray(wf, dtype=np.float64)

    front = wf[:n_baseline_samples]
    back = wf[-n_baseline_samples:]

    front_rms = _window_rms(front)
    back_rms = _window_rms(back)

    front_ok = front_rms < rms_threshold
    back_ok = back_rms < rms_threshold

    if front_ok and back_ok:
        # both clean -> average them
        bm = 0.5 * (_window_mean(front) + _window_mean(back))
        br = 0.5 * (front_rms + back_rms)
        noisy = False
    elif front_ok and not back_ok:
        bm = _window_mean(front)
        br = front_rms
        noisy = False
    elif back_ok and not front_ok:
        bm = _window_mean(back)
        br = back_rms
        noisy = False
    else:
        bm = np.nan
        br = np.nan
        noisy = True

    return bm, br, noisy


def format_root_data(_ar, n_baseline_samples=50, dac_to_pe=(1.0, 1.0), rms_threshold=0.01):
    time = _ar["Time"]
    wf0 = _ar["wf0"]
    wf1 = _ar["wf1"]

    scales = np.asarray(dac_to_pe, dtype=np.float64)
    if scales.shape != (2,):
        raise ValueError("dac_to_pe must have exactly 2 values, one per channel")

    n = time.shape[0]
    n_samples = wf0.shape[1]
    print("Number of events:", n)
    print("Sample length:", n_samples)

    data = np.empty(
        n,
        dtype=[
            ("event_index", np.int32),
            ("event_time", np.float64),
            ("baseline_mean", np.float32, (2,)),
            ("baseline_rms", np.float32, (2,)),
            ("noisy_baseline", np.bool_, (2,)),
            ("wfs_raw", np.float64, (2, n_samples)),
            ("wfs", np.float64, (2, n_samples)),
        ],
    )

    data["event_index"] = np.arange(n)
    data["event_time"] = time
    data["wfs_raw"] = np.stack([wf0, wf1], axis=1).astype(np.float64)
    data["wfs_raw"] *= scales[None, :, None]

    baseline_mean = np.full((n, 2), np.nan, dtype=np.float64)
    baseline_rms = np.full((n, 2), np.nan, dtype=np.float64)
    noisy_baseline = np.zeros((n, 2), dtype=np.bool_)

    for i in range(n):
        for ch in range(2):
            bm, br, noisy = _estimate_baseline_strict(
                data["wfs_raw"][i, ch, :],
                n_baseline_samples=n_baseline_samples,
                rms_threshold=rms_threshold,
            )
            baseline_mean[i, ch] = bm
            baseline_rms[i, ch] = br
            noisy_baseline[i, ch] = noisy

    # Impute NaNs with the average of all valid baselines per channel
    # nans are missed when baseline is too noisy. Happens with very active pulses.
    for ch in range(2):
        valid = np.isfinite(baseline_mean[:, ch])
        if np.any(valid):
            mean_fill = float(np.mean(baseline_mean[valid, ch]))
            rms_fill = float(np.mean(baseline_rms[valid, ch]))
        else:
            mean_fill = 0.0
            rms_fill = 0.0

        missing = ~valid
        baseline_mean[missing, ch] = mean_fill
        baseline_rms[missing, ch] = rms_fill

    data["baseline_mean"] = baseline_mean.astype(np.float32)
    data["baseline_rms"] = baseline_rms.astype(np.float32)
    data["noisy_baseline"] = noisy_baseline

    # Baseline-subtracted and inverted so pulses are positive
    data["wfs"] = -(data["wfs_raw"] - data["baseline_mean"][:, :, None])

    return data