"""
compute_midi_metrics.py

Quick MIDI-level evaluation script.

Usage:
  python eval/compute_midi_metrics.py --refs path/to/ref_midis --gens path/to/gen_midis --out results.csv
  or
  python eval/compute_midi_metrics.py --auto-dir path/to/parent_of_generated_samples

Notes:
- Matches files by basename (ref: a.mid, gen: a.mid) or by pairing <base>_GT.mid with <base>.mid under --auto-dir.
- In --auto-dir mode, pairing is done inside each folder first, then all pairs are aggregated into one CSV.
- Computes: onset precision/recall/f1, pitch accuracy (on matched onsets), note density (notes/sec), duration,
  pitch JSD, onset JSD.
- Optionally computes feature-based FMD (Fréchet MIDI Distance) for global set and each run_dir.
- If FMD is enabled, bootstrap CI is computed together automatically.
- Outputs CSV and a summary CSV (same dirname): <out> and <out>.summary.csv
- Also outputs grouped summary by folder: <out>.by_run.csv
- If FMD enabled, writes: <out>.fmd.csv
- Requires: pretty_midi, numpy
"""

import argparse
import os
import csv

import numpy as np
import pretty_midi

from eval_acc_jsd_bridge import compute_eval_acc_style_jsd


def load_notes_and_tempo(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []  # tuples (onset, pitch)
    for inst in pm.instruments:
        for n in inst.notes:
            notes.append((n.start, n.pitch))
    tempo_times, tempi = pm.get_tempo_changes()
    bpm = float(tempi[0]) if len(tempi) > 0 else 120.0
    if len(notes) == 0:
        return np.array([], dtype=float), np.array([], dtype=int), 0.0, bpm
    notes = sorted(notes, key=lambda x: x[0])
    onsets = np.array([n[0] for n in notes], dtype=float)
    pitches = np.array([n[1] for n in notes], dtype=int)
    duration = pm.get_end_time()
    return onsets, pitches, duration, bpm


def load_notes(midi_path):
    onsets, pitches, duration, _ = load_notes_and_tempo(midi_path)
    return onsets, pitches, duration


def compute_eval_acc_jsd(ref_path, gen_path):
    """Compute pitch/onset JSD using the canonical eval_acc.py definitions."""
    return compute_eval_acc_style_jsd(ref_path, gen_path)


def match_onsets(ref_onsets, est_onsets, tol=0.05):
    """Greedy matching: for each est, find nearest unmatched ref within tol."""
    if len(ref_onsets) == 0 or len(est_onsets) == 0:
        return 0, []

    ref_used = np.zeros(len(ref_onsets), dtype=bool)
    matches = []  # list of (ref_idx, est_idx)
    for ei, e in enumerate(est_onsets):
        diffs = np.abs(ref_onsets - e)
        cand_idx = np.where((diffs <= tol) & (~ref_used))[0]
        if cand_idx.size > 0:
            best = cand_idx[np.argmin(diffs[cand_idx])]
            ref_used[best] = True
            matches.append((best, ei))
    return len(matches), matches


def onset_metrics(ref_onsets, est_onsets, tol=0.05):
    matches, _ = match_onsets(ref_onsets, est_onsets, tol=tol)
    p = matches / len(est_onsets) if len(est_onsets) > 0 else 0.0
    r = matches / len(ref_onsets) if len(ref_onsets) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, matches


def pitch_accuracy(ref_onsets, ref_pitches, est_onsets, est_pitches, tol=0.05):
    matches, pairs = match_onsets(ref_onsets, est_onsets, tol=tol)
    if matches == 0:
        return 0.0, 0
    correct = 0
    for ri, ei in pairs:
        if est_pitches[ei] == ref_pitches[ri]:
            correct += 1
    acc = correct / matches
    return acc, matches


def normalized_hist(hist):
    hist = np.asarray(hist, dtype=float)
    total = float(np.sum(hist))
    if total <= 0:
        return np.zeros_like(hist, dtype=float)
    return hist / total


def jsd_from_hists(p_hist, q_hist, eps=1e-12):
    p = normalized_hist(p_hist)
    q = normalized_hist(q_hist)
    if np.sum(p) == 0 and np.sum(q) == 0:
        return 0.0

    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    p /= np.sum(p)
    q /= np.sum(q)
    m = 0.5 * (p + q)

    kl_pm = float(np.sum(p * np.log2(p / m)))
    kl_qm = float(np.sum(q * np.log2(q / m)))
    return 0.5 * (kl_pm + kl_qm)


def pitch_histogram(pitches):
    if len(pitches) == 0:
        return np.zeros(88, dtype=float)
    clipped = np.clip(np.asarray(pitches, dtype=int), 21, 108) - 21
    return np.bincount(clipped, minlength=88).astype(float)


def onset_phase_histogram(onsets, bpm, bins=16):
    if len(onsets) == 0:
        return np.zeros(bins, dtype=float)
    bpm = max(float(bpm), 1e-8)
    sec_per_beat = 60.0 / bpm
    phases = (np.asarray(onsets, dtype=float) / sec_per_beat) % 1.0
    hist, _ = np.histogram(phases, bins=bins, range=(0.0, 1.0))
    return hist.astype(float)


def eval_pair(ref_path, gen_path, tol=0.05):
    ref_onsets, ref_pitches, ref_dur, ref_bpm = load_notes_and_tempo(ref_path)
    gen_onsets, gen_pitches, gen_dur, gen_bpm = load_notes_and_tempo(gen_path)

    p, r, f1, matched = onset_metrics(ref_onsets, gen_onsets, tol=tol)
    pitch_acc, matched_count = pitch_accuracy(ref_onsets, ref_pitches, gen_onsets, gen_pitches, tol=tol)

    note_density_ref = len(ref_onsets) / ref_dur if ref_dur > 0 else 0.0
    note_density_gen = len(gen_onsets) / gen_dur if gen_dur > 0 else 0.0
    pitch_jsd, onset_jsd = compute_eval_acc_jsd(ref_path, gen_path)

    return {
        "ref_file": os.path.basename(ref_path),
        "gen_file": os.path.basename(gen_path),
        "ref_notes": len(ref_onsets),
        "gen_notes": len(gen_onsets),
        "ref_dur": ref_dur,
        "gen_dur": gen_dur,
        "onset_p": p,
        "onset_r": r,
        "onset_f1": f1,
        "pitch_acc": pitch_acc,
        "pitch_jsd": pitch_jsd,
        "onset_jsd": onset_jsd,
        "matched": matched_count,
        "note_density_ref": note_density_ref,
        "note_density_gen": note_density_gen,
    }


def extract_midi_feature_vector(midi_path):
    """Extract a compact feature vector from MIDI for feature-based FMD."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in pm.instruments:
        notes.extend(inst.notes)

    # 11 scalar features + 12 pitch-class histogram bins
    feat_dim = 23
    if len(notes) == 0:
        return np.zeros(feat_dim, dtype=float)

    starts = np.array([n.start for n in notes], dtype=float)
    pitches = np.array([n.pitch for n in notes], dtype=float)
    velocities = np.array([n.velocity for n in notes], dtype=float)
    durs = np.array([max(0.0, n.end - n.start) for n in notes], dtype=float)
    midi_dur = max(pm.get_end_time(), 1e-8)

    starts_sorted = np.sort(starts)
    if starts_sorted.size >= 2:
        ioi = np.diff(starts_sorted)
        ioi_mean = float(np.mean(ioi))
        ioi_std = float(np.std(ioi))
    else:
        ioi_mean = 0.0
        ioi_std = 0.0

    pitch_class = (pitches.astype(int) % 12)
    pch = np.bincount(pitch_class, minlength=12).astype(float)
    pch /= max(np.sum(pch), 1.0)

    scalar = np.array(
        [
            np.log1p(len(notes)),
            np.log1p(midi_dur),
            len(notes) / midi_dur,
            float(np.mean(pitches)),
            float(np.std(pitches)),
            float(np.mean(velocities) / 127.0),
            float(np.std(velocities) / 127.0),
            float(np.mean(durs)),
            float(np.std(durs)),
            ioi_mean,
            ioi_std,
        ],
        dtype=float,
    )
    return np.concatenate([scalar, pch], axis=0)


def features_to_mu_cov(features):
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[0] == 0:
        return None, None
    mu = np.mean(x, axis=0)
    if x.shape[0] == 1:
        cov = np.zeros((x.shape[1], x.shape[1]), dtype=float)
    else:
        cov = np.cov(x, rowvar=False, ddof=0)
    return mu, cov


def _sqrtm_psd(mat):
    """Matrix square root for symmetric PSD matrix."""
    mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    """Fréchet distance between two Gaussians."""
    if mu1 is None or mu2 is None:
        return None

    d = mu1.shape[0]
    eye = np.eye(d, dtype=float)
    cov1 = cov1 + eps * eye
    cov2 = cov2 + eps * eye

    diff = mu1 - mu2
    sqrt_cov1 = _sqrtm_psd(cov1)
    middle = sqrt_cov1 @ cov2 @ sqrt_cov1
    middle = 0.5 * (middle + middle.T)
    eigvals = np.linalg.eigvalsh(middle)
    eigvals = np.clip(eigvals, 0.0, None)
    trace_sqrt = float(np.sum(np.sqrt(eigvals)))

    fd = float(diff @ diff + np.trace(cov1) + np.trace(cov2) - 2.0 * trace_sqrt)
    if fd < 0 and abs(fd) < 1e-9:
        fd = 0.0
    return fd


def bootstrap_fmd_ci(ref_features, gen_features, n_bootstrap=300, ci_level=0.95, seed=1234):
    """Bootstrap CI for FMD by resampling ref/gen features independently with replacement."""
    x_ref = np.asarray(ref_features, dtype=float)
    x_gen = np.asarray(gen_features, dtype=float)
    if x_ref.ndim != 2 or x_gen.ndim != 2 or x_ref.shape[0] < 2 or x_gen.shape[0] < 2:
        return None, None
    if n_bootstrap <= 0:
        return None, None

    rng = np.random.default_rng(seed)
    vals = []
    n_ref = x_ref.shape[0]
    n_gen = x_gen.shape[0]
    for _ in range(n_bootstrap):
        idx_r = rng.integers(0, n_ref, size=n_ref)
        idx_g = rng.integers(0, n_gen, size=n_gen)
        mu_r, cov_r = features_to_mu_cov(x_ref[idx_r])
        mu_g, cov_g = features_to_mu_cov(x_gen[idx_g])
        fd = frechet_distance(mu_r, cov_r, mu_g, cov_g)
        if fd is not None and np.isfinite(fd):
            vals.append(fd)

    if len(vals) == 0:
        return None, None

    alpha = 1.0 - ci_level
    low = float(np.quantile(vals, alpha / 2.0))
    high = float(np.quantile(vals, 1.0 - alpha / 2.0))
    return low, high


def find_pairs(ref_dir, gen_dir, match_gt_suffix=False):
    # If match_gt_suffix is True, look for files like NAME_GT.mid and NAME.mid in the same folder
    if match_gt_suffix:
        files = os.listdir(ref_dir)
        pairs = []
        for f in files:
            if f.endswith("_GT.mid"):
                base_no_ext = os.path.splitext(f)[0]
                base = base_no_ext[:-3]  # strip trailing '_GT'
                ref_path = os.path.join(ref_dir, f)
                gen_name = base + ".mid"
                gen_path = os.path.join(gen_dir, gen_name)
                if os.path.exists(gen_path):
                    pairs.append(
                        {
                            "ref_path": ref_path,
                            "gen_path": gen_path,
                            "pair_id": base,
                            "run_dir": "",
                        }
                    )
        return pairs

    refs = {os.path.splitext(f)[0]: os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith(".mid")}
    gens = {os.path.splitext(f)[0]: os.path.join(gen_dir, f) for f in os.listdir(gen_dir) if f.endswith(".mid")}
    common = sorted(set(refs.keys()) & set(gens.keys()))
    pairs = [
        {
            "ref_path": refs[k],
            "gen_path": gens[k],
            "pair_id": k,
            "run_dir": "",
        }
        for k in common
    ]
    return pairs


def find_pairs_auto(root_dir, gt_suffix="_GT"):
    """Recursively scan `root_dir` and pair files where one file is named <base>_GT.mid and the other <base>.mid.

    Pairing is done inside each folder to avoid cross-folder name collisions.
    Returns list of dicts: {ref_path, gen_path, pair_id, run_dir}.
    """
    pairs = []
    root_dir = os.path.abspath(root_dir)

    for dpath, _, files in os.walk(root_dir):
        local_refs = {}
        local_gens = {}

        for f in files:
            if not f.lower().endswith(".mid"):
                continue
            full = os.path.join(dpath, f)
            stem = os.path.splitext(f)[0]
            if stem.endswith(gt_suffix):
                base = stem[: -len(gt_suffix)]
                local_refs[base] = full
            else:
                local_gens[stem] = full

        common = sorted(set(local_refs.keys()) & set(local_gens.keys()))
        run_dir = os.path.relpath(dpath, root_dir)
        if run_dir == ".":
            run_dir = ""

        for base in common:
            pairs.append(
                {
                    "ref_path": local_refs[base],
                    "gen_path": local_gens[base],
                    "pair_id": base,
                    "run_dir": run_dir,
                }
            )
    return pairs


def build_summary(rows, numeric_keys):
    summary = {}
    for k in numeric_keys:
        vals = np.array([float(row.get(k, 0.0)) for row in rows], dtype=float)
        summary[k] = {
            "count": int(vals.size),
            "mean": float(np.mean(vals)) if vals.size > 0 else None,
            "std": float(np.std(vals, ddof=0)) if vals.size > 0 else None,
            "var": float(np.var(vals, ddof=0)) if vals.size > 0 else None,
            "median": float(np.median(vals)) if vals.size > 0 else None,
            "min": float(np.min(vals)) if vals.size > 0 else None,
            "max": float(np.max(vals)) if vals.size > 0 else None,
        }
    return summary


def format_sig(value, sig_digits=4):
    """Format numeric value to N significant digits for CSV readability."""
    if value is None or value == "":
        return ""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        if np.isnan(value) or np.isinf(value):
            return str(value)
        return f"{float(value):.{sig_digits}g}"
    return value


def format_row_for_csv(row, keys, numeric_key_set, sig_digits):
    out = {}
    for k in keys:
        v = row.get(k, "")
        if k in numeric_key_set:
            out[k] = format_sig(v, sig_digits=sig_digits)
        else:
            out[k] = v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refs", required=False, help="dir of reference MIDI files")
    parser.add_argument("--gens", required=False, help="dir of generated MIDI files")
    parser.add_argument(
        "--auto-dir", required=False, help="root dir to recursively scan for paired *_GT.mid and .mid files"
    )
    parser.add_argument("--gt-suffix", default="_GT", help="suffix used to mark GT files (default: _GT)")
    parser.add_argument("--out", default="eval_results.csv")
    parser.add_argument("--match-gt-suffix", action="store_true", help="match NAME_GT.mid with NAME.mid in same folder")
    parser.add_argument("--tol", type=float, default=0.05, help="onset tolerance in seconds")
    parser.add_argument("--sig-digits", type=int, default=4, help="significant digits in CSV output (default: 4)")
    parser.add_argument(
        "--compute-fmd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="compute feature-based FMD globally and by run_dir",
    )
    parser.add_argument("--fmd-bootstrap-iters", type=int, default=300, help="bootstrap iterations for FMD CI")
    parser.add_argument("--fmd-ci-level", type=float, default=0.95, help="confidence level for FMD CI")
    parser.add_argument("--fmd-bootstrap-seed", type=int, default=1234, help="seed for FMD bootstrap")
    args = parser.parse_args()

    if args.sig_digits <= 0:
        print("--sig-digits must be > 0")
        return
    if args.fmd_bootstrap_iters < 0:
        print("--fmd-bootstrap-iters must be >= 0")
        return
    if not (0.0 < args.fmd_ci_level < 1.0):
        print("--fmd-ci-level must be in (0, 1)")
        return

    # Auto-scan mode: if --auto-dir provided, find pairs under that root
    if args.auto_dir:
        pairs = find_pairs_auto(args.auto_dir, gt_suffix=args.gt_suffix)
    else:
        if not args.refs or not args.gens:
            print("Either --auto-dir or both --refs and --gens must be provided.")
            return
        pairs = find_pairs(args.refs, args.gens, match_gt_suffix=args.match_gt_suffix)

    if len(pairs) == 0:
        print("No matching MIDI filenames found.")
        return

    # If using --auto-dir and user didn't supply an explicit --out, set
    # out to out/<last_dir>/eval_results.csv for convenience.
    if args.auto_dir and (args.out is None or args.out == "eval_results.csv"):
        last_dir = os.path.basename(os.path.normpath(args.auto_dir))
        out_dir = os.path.join("out", last_dir)
        os.makedirs(out_dir, exist_ok=True)
        args.out = os.path.join(out_dir, "eval_results.csv")

    rows = []
    global_ref_features = []
    global_gen_features = []
    run_ref_features = {}
    run_gen_features = {}
    for pair in pairs:
        ref_path = pair["ref_path"]
        gen_path = pair["gen_path"]
        print(f"Evaluating {os.path.basename(ref_path)}")
        r = eval_pair(ref_path, gen_path, tol=args.tol)
        r["pair_id"] = pair.get("pair_id", "")
        r["run_dir"] = pair.get("run_dir", "")
        r["ref_path"] = ref_path
        r["gen_path"] = gen_path
        rows.append(r)

        if args.compute_fmd:
            rf = extract_midi_feature_vector(ref_path)
            gf = extract_midi_feature_vector(gen_path)
            global_ref_features.append(rf)
            global_gen_features.append(gf)

            run_dir = pair.get("run_dir", "") or "."
            run_ref_features.setdefault(run_dir, []).append(rf)
            run_gen_features.setdefault(run_dir, []).append(gf)

    # write CSV
    keys = [
        "run_dir",
        "pair_id",
        "ref_path",
        "gen_path",
        "ref_file",
        "gen_file",
        "ref_notes",
        "gen_notes",
        "ref_dur",
        "gen_dur",
        "onset_p",
        "onset_r",
        "onset_f1",
        "pitch_acc",
        "pitch_jsd",
        "onset_jsd",
        "matched",
        "note_density_ref",
        "note_density_gen",
    ]
    numeric_keys = [
        "ref_notes",
        "gen_notes",
        "ref_dur",
        "gen_dur",
        "onset_p",
        "onset_r",
        "onset_f1",
        "pitch_acc",
        "pitch_jsd",
        "onset_jsd",
        "matched",
        "note_density_ref",
        "note_density_gen",
    ]
    numeric_key_set = set(numeric_keys)

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(format_row_for_csv(row, keys, numeric_key_set, sig_digits=args.sig_digits))

    summary = build_summary(rows, numeric_keys)

    # also keep simple averages for quick print
    quick_avg = {
        k: summary[k]["mean"]
        for k in ["onset_p", "onset_r", "onset_f1", "pitch_acc", "pitch_jsd", "onset_jsd"]
    }
    print("Averages:")
    print({k: format_sig(v, sig_digits=args.sig_digits) for k, v in quick_avg.items()})

    # write a human-readable summary CSV next to the CSV (easier to open)
    try:
        summary_csv = args.out + ".summary.csv"
        with open(summary_csv, "w", newline="") as scf:
            swriter = csv.writer(scf)
            # header
            swriter.writerow(["field", "count", "mean", "std", "var", "median", "min", "max"])
            for k in numeric_keys:
                v = summary.get(k, {})
                swriter.writerow(
                    [
                        k,
                        v.get("count", ""),
                        format_sig(v.get("mean", ""), sig_digits=args.sig_digits),
                        format_sig(v.get("std", ""), sig_digits=args.sig_digits),
                        format_sig(v.get("var", ""), sig_digits=args.sig_digits),
                        format_sig(v.get("median", ""), sig_digits=args.sig_digits),
                        format_sig(v.get("min", ""), sig_digits=args.sig_digits),
                        format_sig(v.get("max", ""), sig_digits=args.sig_digits),
                    ]
                )

        print(f"Wrote summary CSV to {summary_csv}")
    except Exception as e:
        print("Failed writing summary:", e)

    # write grouped summary by run_dir (useful for offline-0318 style folders)
    try:
        by_run = {}
        for row in rows:
            run_dir = row.get("run_dir", "") or "."
            by_run.setdefault(run_dir, []).append(row)

        by_run_csv = args.out + ".by_run.csv"
        with open(by_run_csv, "w", newline="") as bf:
            bw = csv.writer(bf)
            bw.writerow(
                [
                    "run_dir",
                    "num_pairs",
                    "onset_p_mean",
                    "onset_r_mean",
                    "onset_f1_mean",
                    "pitch_acc_mean",
                    "pitch_jsd_mean",
                    "onset_jsd_mean",
                    "matched_mean",
                    "note_density_ref_mean",
                    "note_density_gen_mean",
                    "fmd",
                    "fmd_ci_low",
                    "fmd_ci_high",
                ]
            )
            for run_dir in sorted(by_run.keys()):
                group_rows = by_run[run_dir]
                group_summary = build_summary(group_rows, numeric_keys)
                run_fmd = ""
                run_ci_low = ""
                run_ci_high = ""
                if args.compute_fmd:
                    ref_feats = run_ref_features.get(run_dir, [])
                    gen_feats = run_gen_features.get(run_dir, [])
                    mu_r, cov_r = features_to_mu_cov(ref_feats)
                    mu_g, cov_g = features_to_mu_cov(gen_feats)
                    run_fmd_val = frechet_distance(mu_r, cov_r, mu_g, cov_g)
                    run_fmd = format_sig(run_fmd_val, sig_digits=args.sig_digits)
                    ci_low, ci_high = bootstrap_fmd_ci(
                        ref_feats,
                        gen_feats,
                        n_bootstrap=args.fmd_bootstrap_iters,
                        ci_level=args.fmd_ci_level,
                        seed=args.fmd_bootstrap_seed,
                    )
                    run_ci_low = format_sig(ci_low, sig_digits=args.sig_digits)
                    run_ci_high = format_sig(ci_high, sig_digits=args.sig_digits)
                bw.writerow(
                    [
                        run_dir,
                        len(group_rows),
                        format_sig(group_summary["onset_p"]["mean"], sig_digits=args.sig_digits),
                        format_sig(group_summary["onset_r"]["mean"], sig_digits=args.sig_digits),
                        format_sig(group_summary["onset_f1"]["mean"], sig_digits=args.sig_digits),
                        format_sig(group_summary["pitch_acc"]["mean"], sig_digits=args.sig_digits),
                        format_sig(group_summary["pitch_jsd"]["mean"], sig_digits=args.sig_digits),
                        format_sig(group_summary["onset_jsd"]["mean"], sig_digits=args.sig_digits),
                        format_sig(group_summary["matched"]["mean"], sig_digits=args.sig_digits),
                        format_sig(group_summary["note_density_ref"]["mean"], sig_digits=args.sig_digits),
                        format_sig(group_summary["note_density_gen"]["mean"], sig_digits=args.sig_digits),
                        run_fmd,
                        run_ci_low,
                        run_ci_high,
                    ]
                )

        print(f"Wrote grouped summary CSV to {by_run_csv}")
    except Exception as e:
        print("Failed writing grouped summary:", e)

    if args.compute_fmd:
        try:
            mu_ref, cov_ref = features_to_mu_cov(global_ref_features)
            mu_gen, cov_gen = features_to_mu_cov(global_gen_features)
            global_fmd = frechet_distance(mu_ref, cov_ref, mu_gen, cov_gen)
            global_ci_low, global_ci_high = bootstrap_fmd_ci(
                global_ref_features,
                global_gen_features,
                n_bootstrap=args.fmd_bootstrap_iters,
                ci_level=args.fmd_ci_level,
                seed=args.fmd_bootstrap_seed,
            )

            fmd_csv = args.out + ".fmd.csv"
            with open(fmd_csv, "w", newline="") as ff:
                fwriter = csv.writer(ff)
                fwriter.writerow(["scope", "run_dir", "num_pairs", "fmd", "ci_low", "ci_high", "ci_level"])
                fwriter.writerow(
                    [
                        "global",
                        ".",
                        len(rows),
                        format_sig(global_fmd, sig_digits=args.sig_digits),
                        format_sig(global_ci_low, sig_digits=args.sig_digits),
                        format_sig(global_ci_high, sig_digits=args.sig_digits),
                        format_sig(args.fmd_ci_level, sig_digits=args.sig_digits),
                    ]
                )

                by_run = {}
                for row in rows:
                    run_dir = row.get("run_dir", "") or "."
                    by_run.setdefault(run_dir, 0)
                    by_run[run_dir] += 1

                for run_dir in sorted(by_run.keys()):
                    ref_feats = run_ref_features.get(run_dir, [])
                    gen_feats = run_gen_features.get(run_dir, [])
                    mu_r, cov_r = features_to_mu_cov(ref_feats)
                    mu_g, cov_g = features_to_mu_cov(gen_feats)
                    run_fmd = frechet_distance(mu_r, cov_r, mu_g, cov_g)
                    ci_low, ci_high = bootstrap_fmd_ci(
                        ref_feats,
                        gen_feats,
                        n_bootstrap=args.fmd_bootstrap_iters,
                        ci_level=args.fmd_ci_level,
                        seed=args.fmd_bootstrap_seed,
                    )
                    fwriter.writerow(
                        [
                            "by_run",
                            run_dir,
                            by_run[run_dir],
                            format_sig(run_fmd, sig_digits=args.sig_digits),
                            format_sig(ci_low, sig_digits=args.sig_digits),
                            format_sig(ci_high, sig_digits=args.sig_digits),
                            format_sig(args.fmd_ci_level, sig_digits=args.sig_digits),
                        ]
                    )

            print(f"FMD(global): {format_sig(global_fmd, sig_digits=args.sig_digits)}")
            print(
                f"FMD CI(global,{int(args.fmd_ci_level*100)}%): "
                f"[{format_sig(global_ci_low, sig_digits=args.sig_digits)}, {format_sig(global_ci_high, sig_digits=args.sig_digits)}]"
            )
            print(f"Wrote FMD CSV to {fmd_csv}")
        except Exception as e:
            print("Failed computing/writing FMD:", e)


if __name__ == "__main__":
    main()
