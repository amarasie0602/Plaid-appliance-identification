"""Feature extraction for PLAID waveforms. Same code for training (02) and prediction."""
import numpy as np
import pandas as pd

FS = 30_000          # samples per second
F0 = 60              # mains frequency (Hz)
SPC = FS // F0       # samples per cycle = 500
N_CYC = 20           # cycles in the analysis window


def load_wave(path):
    w = pd.read_csv(path, header=None, names=["current", "voltage"])
    return w["current"].to_numpy(), w["voltage"].to_numpy()


def rising_crossings(v, min_gap=int(0.8 * SPC)):
    idx = np.flatnonzero((v[:-1] < 0) & (v[1:] >= 0))
    keep = []
    for k in idx:
        if not keep or k - keep[-1] >= min_gap:
            keep.append(k)
    return np.array(keep)


def cycle_rms(i):
    n = len(i) // SPC
    return np.sqrt((i[:n * SPC].reshape(n, SPC) ** 2).mean(axis=1))


def stable_segments(i, win=10, tol=0.02, min_on=0.03, min_len=N_CYC):
    rms = pd.Series(cycle_rms(i))
    cv = rms.rolling(win, center=True).std() / rms.rolling(win, center=True).mean()
    stable = ((cv < tol) & (rms > min_on)).to_numpy().astype(int)
    edges = np.diff(np.r_[0, stable, 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    return [(a, b) for a, b in zip(starts, ends) if b - a >= min_len]


def steady_window_start(i, v, n_cycles=N_CYC):
    segs = stable_segments(i)
    if not segs:
        return None
    a, b = max(segs, key=lambda s: s[1] - s[0])
    r = rising_crossings(v)
    ok = r[(r >= a * SPC) & (r + n_cycles * SPC <= b * SPC)]
    return ok[-1] if len(ok) else None


def prepare(i, v, n_cycles=N_CYC, settle_s=0.5):
    v = v * 120 / np.sqrt(np.mean(v ** 2))
    r = rising_crossings(v)
    start = steady_window_start(i, v, n_cycles)
    stable_found = start is not None
    if start is None:
        start = r[r + n_cycles * SPC <= len(v)][-1]
    end = start + n_cycles * SPC
    level = np.sqrt(np.mean(i[start:end] ** 2))
    above = np.flatnonzero(cycle_rms(i) > 0.5 * level)
    onset = above[0] * SPC if len(above) and above[0] > 0 else None
    earliest = onset + int(settle_s * FS) if onset is not None else 0
    return {"i": i, "v": v, "i_win": i[start:end], "v_win": v[start:end],
            "onset": onset, "start": start, "stable_found": stable_found,
            "window_ok": start >= earliest,
            "spike_in_window": np.abs(v[start:end]).max() > 200}


def power_features(i_win, v_win):
    i_rms = np.sqrt(np.mean(i_win ** 2)); v_rms = np.sqrt(np.mean(v_win ** 2))
    P = np.mean(v_win * i_win); S = v_rms * i_rms
    PF = P / S if S > 0 else np.nan
    Q = np.sqrt(max(S ** 2 - P ** 2, 0.0))
    I1 = np.fft.rfft(i_win)[N_CYC]; V1 = np.fft.rfft(v_win)[N_CYC]
    phase = np.degrees(np.angle(I1 * np.conj(V1)))
    return {"i_rms": i_rms, "i_peak": np.abs(i_win).max(), "p_w": P, "q_var": Q,
            "pf": PF, "dpf": np.cos(np.radians(phase)), "phase_deg": phase}


def shape_features(i_win, on_frac=0.05, min_amp=0.03):
    peak = np.abs(i_win).max(); rms = np.sqrt(np.mean(i_win ** 2))
    z = (i_win - i_win.mean()) / i_win.std()
    cycles = i_win.reshape(N_CYC, SPC); avg = cycles.mean(axis=0)
    return {"crest_factor": peak / rms, "form_factor": rms / np.mean(np.abs(i_win)),
            "skewness": np.mean(z ** 3), "kurtosis": np.mean(z ** 4) - 3,
            "conduction_angle": np.mean(np.abs(i_win) > max(on_frac * peak, min_amp)),
            "cycle_stability": np.mean([np.corrcoef(c, avg)[0, 1] for c in cycles])}


def harmonic_features(i_win, n_max=40, hf_hz=2400):
    X = np.abs(np.fft.rfft(i_win)); h = X[N_CYC::N_CYC][:n_max]
    hf_bin = int(hf_hz * len(i_win) / FS)
    return {"thd": np.sqrt(np.sum(h[1:] ** 2)) / h[0],
            "h2_h1": h[1] / h[0], "h3_h1": h[2] / h[0], "h5_h1": h[4] / h[0],
            "h7_h1": h[6] / h[0], "h9_h1": h[8] / h[0],
            "hf_noise_ratio": np.sqrt(np.sum(X[hf_bin + 1:] ** 2) / np.sum(X[1:] ** 2)),
            "half_cycle_asym": abs(np.mean(i_win)) / np.mean(np.abs(i_win))}


def vi_features(i_win, v_win):
    i_c = i_win.reshape(N_CYC, SPC).mean(axis=0); v_c = v_win.reshape(N_CYC, SPC).mean(axis=0)
    i_n, v_n = i_c / np.abs(i_c).max(), v_c / np.abs(v_c).max()
    area = 0.5 * abs(np.sum(v_n * np.roll(i_n, -1) - np.roll(v_n, -1) * i_n))
    near0 = np.abs(v_n) < 0.05
    mid, end = np.abs(v_n) < 0.3, np.abs(v_n) > 0.7
    s_mid = np.polyfit(v_n[mid], i_n[mid], 1)[0]; s_end = np.polyfit(v_n[end], i_n[end], 1)[0]
    return {"vi_area": area, "vi_width0": np.ptp(i_n[near0]) if near0.any() else np.nan,
            "vi_mid_slope": s_mid, "vi_curvature": s_end / s_mid if abs(s_mid) > 0.3 else np.nan,
            "v_at_ipeak": abs(v_n[np.argmax(np.abs(i_n))])}


def transient_features(p, tol=0.05, high=1.5):
    nan = {"inrush_ratio": np.nan, "settling_time": np.nan, "transient_energy": np.nan, "inrush_duration": np.nan}
    if p["onset"] is None:
        return nan
    i, i_win = p["i"], p["i_win"]
    level = np.sqrt(np.mean(i_win ** 2))
    on, st = p["onset"] // SPC, p["start"] // SPC
    rel = cycle_rms(i)[on:st] / level
    outside = np.flatnonzero(np.abs(rel - 1) > tol)
    settle = outside[-1] + 1 if len(outside) else 0
    return {"inrush_ratio": np.abs(i[p["onset"]:]).max() / np.abs(i_win).max(),
            "settling_time": settle / F0,
            "transient_energy": np.clip(rel[:settle] - 1, 0, None).sum() / F0,
            "inrush_duration": np.sum(rel > high) / F0}


def extract_features(i, v):
    p = prepare(i, v)
    feats = {}
    feats.update(power_features(p["i_win"], p["v_win"]))
    feats.update(shape_features(p["i_win"]))
    feats.update(harmonic_features(p["i_win"]))
    feats.update(vi_features(p["i_win"], p["v_win"]))
    feats.update(transient_features(p))
    feats.update({"stable_found": p["stable_found"], "window_ok": p["window_ok"],
                  "spike_in_window": p["spike_in_window"]})
    return feats
