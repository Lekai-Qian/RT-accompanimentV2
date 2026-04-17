"""Utilities to compare generated accompaniment MIDI files against ground-truth accompaniment.

The script expects two folders:
- one containing generated outputs that bundle melody and accompaniment tracks
- another containing ground-truth accompaniment-only MIDI files

For every basename shared by those folders, the script computes Jensen-Shannon
Divergence (JSD) between the generated accompaniment and the ground truth across
three feature distributions: pitch, onset position, and note duration. It also
reports a simple harmonic consonance ratio between the generated melody and
accompaniment tracks to estimate how often the two align harmonically.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import re
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
import os
import shutil
import tempfile
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pretty_midi
import mir_eval.chord


LOGGER = logging.getLogger(__name__)


POLYDIS_CACHE: dict[Path, dict] = {}

_ACCOMPANIMENT_TRACK_NAMES = {"piano"}


@dataclass
class DistributionConfig:
    pitch_bins: Sequence[int]
    onset_bins: np.ndarray
    duration_bins: np.ndarray


@dataclass
class ChordAccuracyConfig:
    min_total_duration: float = 0.05
    coverage_threshold: float = 0.2
    penalty_lambda: float = 0.5
    root_bonus: float = 0.3


@dataclass
class PhraseMetrics:
    song: str
    index: int
    rhythm_density: float
    voice_number: float
    start: Optional[float] = None
    end: Optional[float] = None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate generated accompaniment against ground truth MIDI.")
    parser.add_argument("--generated-dir", type=Path, required=True, help="Directory with generated MIDI files.")
    parser.add_argument(
        "--groundtruth-dir", type=Path, required=True, help="Directory with ground-truth accompaniment MIDI files."
    )
    parser.add_argument(
        "--chord-annotation-root",
        type=Path,
        default=None,
        help="Root directory containing chord annotations (subdirectories named after piece ids with chord_midi.txt).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path where per-piece metrics (including histograms) are stored as JSON.",
    )
    parser.add_argument(
        "--validation-phrases",
        type=Path,
        default=None,
        help="Optional JSONL file containing validation accompaniment phrases for RD/VN consistency analysis.",
    )
    parser.add_argument(
        "--pair-analysis-random",
        type=int,
        default=1000,
        help="Number of random phrase pairs sampled when estimating RD/VN consistency (ignored if validation phrases not provided).",
    )
    parser.add_argument(
        "--phrase-window-seconds",
        type=float,
        default=2.0,
        help="Window length in seconds when segmenting validation phrases for RD/VN analysis.",
    )
    parser.add_argument(
        "--phrase-rhythm-resolution",
        type=float,
        default=1e-2,
        help="Temporal resolution (seconds) when computing rhythm density inside each window.",
    )
    parser.add_argument(
        "--auto-phrase-analysis",
        action="store_true",
        help="Automatically segment MIDI into fixed bar windows for RD/VN analysis when validation phrases are not provided.",
    )
    parser.add_argument(
        "--phrase-bars",
        type=int,
        default=4,
        help="Number of bars per window used during automatic phrase analysis.",
    )
    parser.add_argument(
        "--polydis-root",
        type=Path,
        default=None,
        help="Optional root directory of the PolyDis codebase (expects poly_dis/model_param/polydis-v1.pt) to enable latent similarity metrics.",
    )
    parser.add_argument(
        "--frechet-music-distance",
        action="store_true",
        help="Compute Frechet Music Distance (FMD) between generated accompaniment and ground-truth accompaniment.",
    )
    parser.add_argument(
        "--frechet-model",
        default="clamp2",
        choices=("clamp2", "clamp"),
        help="Embedding model used for Frechet Music Distance (default: clamp2).",
    )
    parser.add_argument(
        "--frechet-estimator",
        default="mle",
        choices=("mle", "bootstrap", "oas", "shrinkage", "leodit_wolf"),
        help="Gaussian estimator used for Frechet Music Distance (default: mle).",
    )
    parser.add_argument(
        "--frechet-cache-dir",
        type=Path,
        default=None,
        help="Optional directory for caching Frechet Music Distance features.",
    )
    parser.add_argument(
        "--frechet-verbose",
        action="store_true",
        help="Enable verbose logging from Frechet Music Distance feature extractor.",
    )
    parser.add_argument(
        "--frechet-keep-temp",
        action="store_true",
        help="Keep temporary accompaniment-only MIDI files prepared for Frechet Music Distance.",
    )
    parser.add_argument(
        "--melody-track-names",
        nargs="*",
        default=("melody",),
        help="Instrument names (case-insensitive) to treat as melody and exclude from accompaniment in generated files.",
    )
    parser.add_argument(
        "--melody-programs",
        nargs="*",
        type=int,
        default=(),
        help="Program numbers that identify melody instruments to exclude from accompaniment in generated files.",
    )
    parser.add_argument(
        "--melody-track-indices",
        nargs="*",
        type=int,
        default=(),
        help="Instrument indices (0-based) to treat as melody and exclude from accompaniment in generated files.",
    )
    parser.add_argument(
        "--keep-melody",
        action="store_true",
        help="Keep melody tracks in the generated accompaniment analysis instead of removing them.",
    )
    parser.add_argument(
        "--include-drums",
        action="store_true",
        help="Include drum tracks when compiling accompaniment notes.",
    )
    parser.add_argument(
        "--consonant-intervals",
        nargs="*",
        type=int,
        default=(0, 3, 4, 7, 8, 9),
        help="Semitone intervals treated as consonant between melody and accompaniment.",
    )
    parser.add_argument(
        "--onset-bins",
        type=int,
        default=64,
        help="Number of uniform bins for the normalized onset histogram.",
    )
    parser.add_argument(
        "--duration-bins",
        type=int,
        default=64,
        help="Number of uniform bins for the duration histogram (after clipping to the specified quantile).",
    )
    parser.add_argument(
        "--duration-clip-quantile",
        type=float,
        default=0.995,
        help="Clip durations above this quantile before histogramming to reduce outlier influence (range 0-1].",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity.",
    )
    return parser


def _collect_accompaniment_notes(
    midi: pretty_midi.PrettyMIDI,
    melody_names: Iterable[str],
    melody_programs: Iterable[int],
    melody_indices: Iterable[int],
    keep_melody: bool,
    include_drums: bool,
) -> List[pretty_midi.Note]:
    """Return a list of accompaniment notes extracted from ``midi``."""
    melody_name_set = {name.lower() for name in melody_names if name}
    melody_program_set = set(melody_programs)
    melody_index_set = set(melody_indices)

    notes: List[pretty_midi.Note] = []
    for idx, instrument in enumerate(midi.instruments):
        if not include_drums and instrument.is_drum:
            continue

        if keep_melody:
            notes.extend(instrument.notes)
            continue

        name = (instrument.name or "").strip().lower()
        is_melody = False
        if melody_name_set and name and name in melody_name_set:
            is_melody = True
        if melody_program_set and instrument.program in melody_program_set:
            is_melody = True
        if melody_index_set and idx in melody_index_set:
            is_melody = True

        if not is_melody:
            notes.extend(instrument.notes)
    return notes


def _collect_melody_notes(
    midi: pretty_midi.PrettyMIDI,
    melody_names: Iterable[str],
    melody_programs: Iterable[int],
    melody_indices: Iterable[int],
    include_drums: bool,
) -> List[pretty_midi.Note]:
    melody_name_set = {name.lower() for name in melody_names if name}
    melody_program_set = set(melody_programs)
    melody_index_set = set(melody_indices)

    notes: List[pretty_midi.Note] = []
    for idx, instrument in enumerate(midi.instruments):
        if not include_drums and instrument.is_drum:
            continue

        name = (instrument.name or "").strip().lower()
        is_melody = False
        if melody_name_set and name and name in melody_name_set:
            is_melody = True
        if melody_program_set and instrument.program in melody_program_set:
            is_melody = True
        if melody_index_set and idx in melody_index_set:
            is_melody = True

        if is_melody:
            notes.extend(instrument.notes)
    return notes


def _compute_pitch_histogram(notes: Sequence[pretty_midi.Note]) -> np.ndarray:
    pitches = [note.pitch for note in notes]
    hist, _ = np.histogram(pitches, bins=np.arange(129), density=False)
    return hist.astype(np.float64)


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return values / total


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-10
    mask = p > 0
    ratio = (p[mask] + eps) / (q[mask] + eps)
    return float(np.sum(p[mask] * np.log(ratio)))


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if p.size == 0 or q.size == 0:
        return float("nan")
    p_norm = _normalize(p)
    q_norm = _normalize(q)
    m = 0.5 * (p_norm + q_norm)
    return 0.5 * _kl_divergence(p_norm, m) + 0.5 * _kl_divergence(q_norm, m)


def _compute_onset_histogram(
    notes: Sequence[pretty_midi.Note],
    piece_length: float,
    bin_edges: np.ndarray,
) -> np.ndarray:
    if piece_length <= 0:
        return np.zeros(bin_edges.size - 1, dtype=np.float64)
    starts = [max(0.0, min(1.0, note.start / piece_length)) for note in notes]
    hist, _ = np.histogram(starts, bins=bin_edges, density=False)
    return hist.astype(np.float64)


def _compute_duration_histogram(
    notes: Sequence[pretty_midi.Note],
    bin_edges: np.ndarray,
) -> np.ndarray:
    durations = [max(0.0, note.end - note.start) for note in notes]
    if not durations:
        return np.zeros(bin_edges.size - 1, dtype=np.float64)
    clipped = np.clip(durations, bin_edges[0], np.nextafter(bin_edges[-1], 0))
    hist, _ = np.histogram(clipped, bins=bin_edges, density=False)
    return hist.astype(np.float64)


def _determine_distribution_bins(
    generated_notes: Sequence[pretty_midi.Note],
    ground_truth_notes: Sequence[pretty_midi.Note],
    onset_bins: int,
    duration_bins: int,
    duration_clip_quantile: float,
) -> tuple[DistributionConfig, float]:
    all_notes = list(generated_notes) + list(ground_truth_notes)
    if all_notes:
        piece_length = max((note.end for note in all_notes), default=0.0)
    else:
        piece_length = 0.0

    onset_edges = np.linspace(0.0, 1.0, onset_bins + 1)

    durations = np.array([max(0.0, note.end - note.start) for note in all_notes], dtype=np.float64)
    if durations.size == 0:
        max_duration = 0.0
    else:
        quantile = np.clip(duration_clip_quantile, 0.0, 1.0)
        max_duration = float(np.quantile(durations, quantile))
        max_duration = max(max_duration, float(durations.max())) if quantile >= 1.0 else max_duration
    if max_duration <= 0:
        max_duration = 1e-3
    duration_edges = np.linspace(0.0, max_duration, duration_bins + 1)

    return (
        DistributionConfig(
            pitch_bins=list(range(129)),
            onset_bins=onset_edges,
            duration_bins=duration_edges,
        ),
        piece_length,
    )


def _load_chord_annotation(path: Path) -> tuple[np.ndarray, List[str]]:
    intervals: List[tuple[float, float]] = []
    labels: List[str] = []
    with path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            start_str, end_str, label = stripped.split("\t")
            intervals.append((float(start_str), float(end_str)))
            labels.append(label)
    if not intervals:
        return np.zeros((0, 2), dtype=np.float64), []
    return np.asarray(intervals, dtype=np.float64), labels


def _build_chord_candidates(labels: Sequence[str]) -> List[tuple[str, int, np.ndarray]]:
    seen: set[str] = set()
    candidates: List[tuple[str, int, np.ndarray]] = []
    for label in labels:
        if label == "N" or label in seen:
            continue
        try:
            root, intervals, _bass = mir_eval.chord.encode(label)
        except mir_eval.chord.InvalidChordException:
            LOGGER.warning("Skipping unsupported chord label %s", label)
            continue
        if root < 0:
            continue
        rotated = mir_eval.chord.rotate_bitmap_to_root(intervals, root).astype(float)
        candidates.append((label, root, rotated))
        seen.add(label)
    return candidates


def _compute_pitch_class_durations(
    notes: Sequence[pretty_midi.Note],
    start: float,
    end: float,
) -> np.ndarray:
    weights = np.zeros(12, dtype=float)
    for note in notes:
        if note.end <= start:
            continue
        if note.start >= end:
            break
        overlap_start = max(start, note.start)
        overlap_end = min(end, note.end)
        if overlap_end <= overlap_start:
            continue
        weights[note.pitch % 12] += overlap_end - overlap_start
    return weights


def _select_chord_label(
    weights: np.ndarray,
    candidates: Sequence[tuple[str, int, np.ndarray]],
    config: ChordAccuracyConfig,
) -> str:
    total = float(weights.sum())
    if total < config.min_total_duration:
        return "N"

    best_label = "N"
    best_score = -np.inf

    for label, root, bitmap in candidates:
        coverage = float(np.dot(weights, bitmap))
        penalty = total - coverage
        score = coverage - config.penalty_lambda * penalty
        if root >= 0:
            score += config.root_bonus * weights[root]
        coverage_ratio = coverage / total if total > 0 else 0.0
        if coverage_ratio < config.coverage_threshold:
            continue
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def _estimate_chord_sequence(
    notes: Sequence[pretty_midi.Note],
    segments: np.ndarray,
    candidates: Sequence[tuple[str, int, np.ndarray]],
    config: ChordAccuracyConfig,
) -> List[str]:
    predictions: List[str] = []
    for start, end in segments:
        if end <= start:
            predictions.append("N")
            continue
        weights = _compute_pitch_class_durations(notes, start, end)
        predictions.append(_select_chord_label(weights, candidates, config))
    return predictions


def _compute_chord_accuracy(
    piece_stem: str,
    accompaniment_notes: Sequence[pretty_midi.Note],
    annotation_root: Optional[Path],
    config: Optional[ChordAccuracyConfig] = None,
) -> Optional[float]:
    if annotation_root is None:
        return None
    chord_path = annotation_root / piece_stem / "chord_midi.txt"
    if not chord_path.exists():
        LOGGER.warning("Chord annotation missing for %s", piece_stem)
        return None
    segments, reference = _load_chord_annotation(chord_path)
    if segments.size == 0 or not reference:
        LOGGER.warning("Chord annotation empty for %s", piece_stem)
        return None
    if not accompaniment_notes:
        LOGGER.warning("No accompaniment notes available for chord accuracy in %s", piece_stem)
        return None

    sorted_notes = sorted(accompaniment_notes, key=lambda note: note.start)
    if not sorted_notes:
        return None
    accompaniment_end = max(note.end for note in sorted_notes)
    tolerance = 1e-3
    valid_indices = [idx for idx, (start, _end) in enumerate(segments) if start < accompaniment_end + tolerance]
    if not valid_indices:
        return None
    truncated_segments = segments[valid_indices]
    truncated_reference = [reference[idx] for idx in valid_indices]
    candidates = _build_chord_candidates(truncated_reference)
    config = config or ChordAccuracyConfig()
    predictions = _estimate_chord_sequence(sorted_notes, truncated_segments, candidates, config)
    if len(predictions) != len(truncated_reference):
        return None
    matches = sum(1 for ref, pred in zip(truncated_reference, predictions) if ref == pred)
    if not truncated_reference:
        return None
    return matches / len(truncated_reference)


def _compute_rhythm_density(
    events: Sequence[dict],
    start: float,
    end: float,
    resolution: float = 1e-3,
) -> float:
    if not events or end <= start or resolution <= 0:
        return 0.0
    window_duration = end - start
    if window_duration <= 0:
        return 0.0

    onset_indices: set[int] = set()
    for event in events:
        onset = float(event.get("time", 0.0))
        if onset < start or onset >= end:
            continue
        # Offset-based bins avoid double counting near-equal onsets while still
        # attributing each rhythmic attack within the window.
        relative = (onset - start) / resolution
        idx = int(math.floor(relative + 1e-9))
        onset_indices.add(idx)
    if not onset_indices:
        return 0.0
    unique_onsets = len(onset_indices)
    return unique_onsets / window_duration


def _compute_voice_number(events: Sequence[dict], start: float, end: float) -> float:
    if not events or end <= start:
        return 0.0
    points: List[tuple[float, int]] = []
    for event in events:
        note_start = float(event.get("time", 0.0))
        duration = max(0.0, float(event.get("duration", 0.0)))
        note_end = note_start + duration
        if note_end <= start or note_start >= end:
            continue
        clipped_start = max(note_start, start)
        clipped_end = min(note_end, end)
        if clipped_end <= clipped_start:
            continue
        points.append((clipped_start, 1))
        points.append((clipped_end, -1))
    if not points:
        return 0.0
    points.sort(key=lambda item: (item[0], item[1]))
    active = 0
    last_time = start
    total_duration = 0.0
    weighted_sum = 0.0
    for time, delta in points:
        elapsed = time - last_time
        if elapsed > 0:
            weighted_sum += active * elapsed
            total_duration += elapsed
        active += delta
        last_time = time
    if last_time < end and active > 0:
        elapsed = end - last_time
        weighted_sum += active * elapsed
        total_duration += elapsed
    if total_duration <= 0:
        return 0.0
    return weighted_sum / total_duration


def _extract_tempo(midi: pretty_midi.PrettyMIDI) -> float:
    times, tempi = midi.get_tempo_changes()
    if tempi.size > 0:
        tempo = float(tempi[0])
    else:
        tempo = float(midi.estimate_tempo())
    if not math.isfinite(tempo) or tempo <= 0:
        tempo = 120.0
    return tempo


def _prepare_polydis_model(polydis_root: Path):
    key = polydis_root.resolve()
    cached = POLYDIS_CACHE.get(key)
    if isinstance(cached, Exception):
        raise cached
    if cached is not None:
        return cached

    if not polydis_root.exists():
        raise FileNotFoundError(f"PolyDis root not found: {polydis_root}")

    if str(polydis_root) not in sys.path:
        sys.path.insert(0, str(polydis_root))

    try:
        import torch  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for PolyDis metrics") from exc

    try:
        from poly_dis.model import PolyDisVAE  # pylint: disable=import-outside-toplevel
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Failed to import PolyDis modules. Ensure polydis-root points to the icm-deep-music-generation repository."
        ) from exc

    model = PolyDisVAE.init_model()
    param_path = polydis_root / "poly_dis" / "model_param" / "polydis-v1.pt"
    if not param_path.exists():
        error = FileNotFoundError(f"PolyDis model weights not found at {param_path}")
        POLYDIS_CACHE[key] = error
        raise error
    model.load_model(str(param_path))
    cached = {
        "model": model,
        "device": model.device,
    }
    POLYDIS_CACHE[key] = cached
    return cached


def _build_polydis_window_features(
    notes: Sequence[pretty_midi.Note],
    window_start: float,
    window_end: float,
    alpha: float,
    steps_per_beat: int,
    num_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    pr = np.zeros((num_steps, 128), dtype=np.float32)
    chord = np.zeros((num_steps // steps_per_beat, 36), dtype=np.float32)

    for note in notes:
        if note.end <= window_start or note.start >= window_end:
            continue
        clipped_start = max(window_start, note.start)
        clipped_end = min(window_end, note.end)
        if clipped_end <= clipped_start:
            continue
        start_idx = int(math.floor((clipped_start - window_start) / alpha))
        end_idx = int(math.ceil((clipped_end - window_start) / alpha))
        start_idx = max(0, min(num_steps - 1, start_idx))
        end_idx = max(start_idx + 1, min(num_steps, end_idx))
        duration_steps = max(1, end_idx - start_idx)
        pitch = int(note.pitch)
        if 0 <= pitch < 128:
            pr[start_idx, pitch] = max(pr[start_idx, pitch], float(duration_steps))

    beat_duration = steps_per_beat * alpha
    num_beats = num_steps // steps_per_beat
    for beat_idx in range(num_beats):
        beat_start = window_start + beat_idx * beat_duration
        beat_end = beat_start + beat_duration
        active = [note for note in notes if note.end > beat_start and note.start < beat_end]
        if not active:
            continue
        durations = np.zeros(12, dtype=np.float32)
        lowest_pitch = None
        chroma = np.zeros(12, dtype=np.float32)
        for note in active:
            overlap_start = max(beat_start, note.start)
            overlap_end = min(beat_end, note.end)
            length = max(0.0, overlap_end - overlap_start)
            if length <= 0.0:
                continue
            pc = note.pitch % 12
            durations[pc] += length
            chroma[pc] = 1.0
            if lowest_pitch is None or note.pitch < lowest_pitch:
                lowest_pitch = note.pitch
        if durations.sum() <= 0.0:
            continue
        root_pc = int(np.argmax(durations))
        bass_pc = int(lowest_pitch % 12) if lowest_pitch is not None else root_pc
        chord_vec = np.zeros(36, dtype=np.float32)
        chord_vec[root_pc] = 1.0
        chord_vec[12:24] = chroma
        chord_vec[24 + bass_pc] = 1.0
        chord[beat_idx] = chord_vec

    return pr, chord


def _compute_polydis_latent_similarity(
    generated_midi: pretty_midi.PrettyMIDI,
    generated_notes: Sequence[pretty_midi.Note],
    ground_truth_midi: pretty_midi.PrettyMIDI,
    ground_truth_notes: Sequence[pretty_midi.Note],
    polydis_root: Optional[Path],
) -> Optional[dict]:
    if polydis_root is None:
        return None
    if not ground_truth_notes:
        return None

    try:
        context = _prepare_polydis_model(polydis_root)
    except Exception as exc:  # pragma: no cover - runtime environment guard
        LOGGER.warning("Skipping PolyDis metric: %s", exc)
        return None

    tempo = _extract_tempo(ground_truth_midi)
    if tempo <= 0:
        tempo = _extract_tempo(generated_midi)
    if tempo <= 0:
        tempo = 120.0
    steps_per_beat = 4
    num_steps = 32
    alpha = 0.25 * 60.0 / tempo
    segment_duration = num_steps * alpha

    gen_end = max(
        generated_midi.get_end_time(),
        max((note.end for note in generated_notes), default=0.0),
    )
    gt_end = max(
        ground_truth_midi.get_end_time(),
        max((note.end for note in ground_truth_notes), default=0.0),
    )
    analysis_end = max(gen_end, gt_end)
    if analysis_end <= 1e-6:
        analysis_end = segment_duration

    segments = max(1, int(math.ceil(analysis_end / segment_duration)))

    gen_pr_list: List[np.ndarray] = []
    gen_chords: List[np.ndarray] = []
    gt_pr_list: List[np.ndarray] = []
    gt_chords: List[np.ndarray] = []

    for idx in range(segments):
        window_start = idx * segment_duration
        window_end = window_start + segment_duration
        gen_pr, gen_chd = _build_polydis_window_features(
            generated_notes,
            window_start,
            window_end,
            alpha,
            steps_per_beat,
            num_steps,
        )
        gt_pr, gt_chd = _build_polydis_window_features(
            ground_truth_notes,
            window_start,
            window_end,
            alpha,
            steps_per_beat,
            num_steps,
        )
        gen_pr_list.append(gen_pr)
        gen_chords.append(gen_chd)
        gt_pr_list.append(gt_pr)
        gt_chords.append(gt_chd)

    import torch  # pylint: disable=import-outside-toplevel

    device = context["device"]
    model = context["model"]

    gen_pr_tensor = torch.from_numpy(np.stack(gen_pr_list)).float().to(device)
    gt_pr_tensor = torch.from_numpy(np.stack(gt_pr_list)).float().to(device)
    gen_chord_tensor = torch.from_numpy(np.stack(gen_chords)).float().to(device)
    gt_chord_tensor = torch.from_numpy(np.stack(gt_chords)).float().to(device)

    with torch.no_grad():
        z_txt_gen = model.txt_encode(gen_pr_tensor)
        z_txt_gt = model.txt_encode(gt_pr_tensor)
        z_chd_gen = model.chd_encode(gen_chord_tensor)
        z_chd_gt = model.chd_encode(gt_chord_tensor)

    txt_dist = torch.norm(z_txt_gen - z_txt_gt, dim=1).cpu().numpy()
    chd_dist = torch.norm(z_chd_gen - z_chd_gt, dim=1).cpu().numpy()

    return {
        "tempo": tempo,
        "segments": segments,
        "segment_duration": segment_duration,
        "txt_mean_distance": float(txt_dist.mean()) if txt_dist.size else None,
        "txt_std_distance": float(txt_dist.std()) if txt_dist.size else None,
        "chd_mean_distance": float(chd_dist.mean()) if chd_dist.size else None,
        "chd_std_distance": float(chd_dist.std()) if chd_dist.size else None,
    }


def _derive_bar_boundaries(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
    end_time = midi.get_end_time()
    if end_time <= 0:
        return np.zeros(1, dtype=np.float64)

    downbeats = midi.get_downbeats()
    if downbeats.size == 0:
        beats = midi.get_beats()
        if beats.size == 0:
            tempo = midi.estimate_tempo()
            if tempo <= 0:
                tempo = 120.0
            beat_duration = 60.0 / tempo
            beats = np.arange(0.0, end_time + beat_duration, beat_duration)
        numerator = 4
        if midi.time_signature_changes:
            numerator = max(1, midi.time_signature_changes[0].numerator)
        step = max(1, numerator)
        downbeats = beats[::step]
    if downbeats.size == 0 or downbeats[0] > 1e-6:
        downbeats = np.concatenate(([0.0], downbeats))
    if downbeats[-1] < end_time - 1e-6:
        downbeats = np.concatenate((downbeats, [end_time]))
    else:
        downbeats[-1] = max(downbeats[-1], end_time)
    return downbeats


def _segment_phrase_metrics_from_midi(
    midi: pretty_midi.PrettyMIDI,
    notes: Sequence[pretty_midi.Note],
    song: str,
    bars_per_window: int,
    resolution: float,
) -> List[PhraseMetrics]:
    if bars_per_window <= 0 or resolution <= 0:
        return []

    boundaries = _derive_bar_boundaries(midi)
    if boundaries.size < 2:
        return []

    segments: List[tuple[float, float]] = []
    last_index = boundaries.size - 1
    step = max(1, bars_per_window)
    for start_idx in range(0, last_index, step):
        end_idx = min(start_idx + step, last_index)
        start_time = float(boundaries[start_idx])
        end_time = float(boundaries[end_idx])
        if end_time <= start_time:
            continue
        segments.append((start_time, end_time))

    metrics: List[PhraseMetrics] = []
    for idx, (segment_start, segment_end) in enumerate(segments):
        overlapping = [
            {
                "time": float(note.start),
                "duration": max(0.0, float(note.end - note.start)),
            }
            for note in notes
            if note.end > segment_start and note.start < segment_end
        ]
        rd = _compute_rhythm_density(overlapping, segment_start, segment_end, resolution)
        vn = _compute_voice_number(overlapping, segment_start, segment_end)
        metrics.append(
            PhraseMetrics(
                song=song,
                index=idx,
                rhythm_density=rd,
                voice_number=vn,
                start=segment_start,
                end=segment_end,
            )
        )
    return metrics


def _load_phrase_metrics(
    jsonl_path: Path,
    window_size: float,
    resolution: float = 1e-2,
) -> dict[str, List[PhraseMetrics]]:
    groups: dict[str, List[PhraseMetrics]] = {}
    if window_size <= 0:
        LOGGER.warning("Invalid phrase window size %.4f; skipping phrase analysis.", window_size)
        return groups
    if resolution <= 0:
        LOGGER.warning("Invalid rhythm resolution %.6f; skipping phrase analysis.", resolution)
        return groups
    with jsonl_path.open() as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            song = Path(record.get("accompaniment_path", "")).stem or "unknown"
            events = record.get("events", [])
            if not events:
                continue
            starts = [float(event.get("time", 0.0)) for event in events]
            durations = [max(0.0, float(event.get("duration", 0.0))) for event in events]
            min_start = min(starts)
            max_end = max(start + duration for start, duration in zip(starts, durations))
            if max_end <= min_start:
                continue
            num_windows = max(1, int(math.ceil((max_end - min_start) / window_size)))
            song_segments = groups.setdefault(song, [])
            for idx in range(num_windows):
                window_start = min_start + idx * window_size
                window_end = window_start + window_size
                overlapping = [
                    event
                    for event, start, duration in zip(events, starts, durations)
                    if start + duration > window_start and start < window_end
                ]
                if not overlapping:
                    continue
                rd = _compute_rhythm_density(overlapping, window_start, window_end, resolution)
                vn = _compute_voice_number(overlapping, window_start, window_end)
                song_segments.append(
                    PhraseMetrics(
                        song=song,
                        index=len(song_segments),
                        rhythm_density=rd,
                        voice_number=vn,
                        start=window_start,
                        end=window_end,
                    )
                )
    return groups


def _sample_random_pairs(
    phrases: Sequence[PhraseMetrics], sample_size: int, rng: random.Random
) -> List[tuple[PhraseMetrics, PhraseMetrics]]:
    total = len(phrases)
    if total < 2 or sample_size <= 0:
        return []
    total_pairs = total * (total - 1) // 2
    if sample_size >= total_pairs:
        return [(first, second) for first, second in combinations(phrases, 2)]
    seen: set[tuple[int, int]] = set()
    pairs: List[tuple[PhraseMetrics, PhraseMetrics]] = []
    while len(pairs) < sample_size:
        i, j = rng.sample(range(total), 2)
        if i > j:
            i, j = j, i
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((phrases[i], phrases[j]))
    return pairs


def _summarize_pair_differences(pairs: Sequence[tuple[PhraseMetrics, PhraseMetrics]]) -> Optional[dict]:
    if not pairs:
        return None
    rd_diffs = []
    vn_diffs = []
    for first, second in pairs:
        rd_diffs.append(abs(first.rhythm_density - second.rhythm_density))
        vn_diffs.append(abs(first.voice_number - second.voice_number))
    return {
        "count": len(pairs),
        "rd": _five_number_summary(rd_diffs),
        "vn": _five_number_summary(vn_diffs),
    }


def _analyze_phrase_pair_consistency(
    groups: dict[str, List[PhraseMetrics]],
    random_pair_limit: int,
) -> dict[str, Optional[dict]]:
    all_phrases = [phrase for phrases in groups.values() for phrase in phrases]
    rng = random.Random(42)
    random_pairs = _sample_random_pairs(all_phrases, random_pair_limit, rng)

    same_song_pairs: List[tuple[PhraseMetrics, PhraseMetrics]] = []
    adjacent_pairs: List[tuple[PhraseMetrics, PhraseMetrics]] = []
    for phrases in groups.values():
        if len(phrases) < 2:
            continue
        same_song_pairs.extend((first, second) for first, second in combinations(phrases, 2))
        adjacent_pairs.extend((phrases[idx], phrases[idx + 1]) for idx in range(len(phrases) - 1))

    return {
        "random": _summarize_pair_differences(random_pairs),
        "same_song": _summarize_pair_differences(same_song_pairs),
        "adjacent": _summarize_pair_differences(adjacent_pairs),
    }


def _compute_harmonic_consonance(
    melody_notes: Sequence[pretty_midi.Note],
    accompaniment_notes: Sequence[pretty_midi.Note],
    consonant_intervals: Sequence[int],
) -> Optional[dict]:
    if not melody_notes or not accompaniment_notes:
        return None

    consonant_set = {interval % 12 for interval in consonant_intervals}
    total = 0.0
    consonant = 0.0
    unsupported = 0.0

    for note in melody_notes:
        duration = max(0.0, note.end - note.start)
        if duration <= 0:
            continue

        overlaps = [other for other in accompaniment_notes if other.end > note.start and other.start < note.end]
        if not overlaps:
            unsupported += duration
            total += duration
            continue

        is_consonant = any((note.pitch - other.pitch) % 12 in consonant_set for other in overlaps)
        if is_consonant:
            consonant += duration
        total += duration

    if total <= 0:
        return None

    consonant_ratio = consonant / total
    unsupported_ratio = unsupported / total
    dissonant_ratio = max(0.0, 1.0 - consonant_ratio - unsupported_ratio)
    return {
        "consonant_ratio": consonant_ratio,
        "dissonant_ratio": dissonant_ratio,
        "unsupported_ratio": unsupported_ratio,
    }


def _load_midi(path: Path) -> pretty_midi.PrettyMIDI:
    try:
        return pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Failed to load MIDI file: {path}") from exc


_MIDI_SUFFIX_PATTERN = re.compile(r"\.mid(?:i)?", re.IGNORECASE)


def _normalized_stem(path: Path) -> str:
    """Return the basename without any suffixes appended after the first MIDI extension."""
    name = path.name
    match = _MIDI_SUFFIX_PATTERN.search(name)
    if match:
        return name[: match.start()]
    return path.stem


def _match_ground_truth(generated_dir: Path, groundtruth_dir: Path) -> List[tuple[Path, Path]]:
    gt_map: dict[str, Path] = {}
    for gt_path in groundtruth_dir.glob("*.mid*"):
        key = _normalized_stem(gt_path)
        if key in gt_map:
            LOGGER.warning("Multiple ground truth files map to key %s; keeping %s", key, gt_map[key].name)
            continue
        gt_map[key] = gt_path
    pairs: List[tuple[Path, Path]] = []
    for gen_path in sorted(generated_dir.glob("*.mid*")):
        key = _normalized_stem(gen_path)
        match = gt_map.get(key)
        if match is None:
            LOGGER.warning("No ground truth found for %s", gen_path.name)
            continue
        pairs.append((gen_path, match))
    return pairs


def _build_accompaniment_only_midi(
    midi: pretty_midi.PrettyMIDI,
    melody_names: Iterable[str],
    melody_programs: Iterable[int],
    melody_indices: Iterable[int],
    keep_melody: bool,
    include_drums: bool,
) -> pretty_midi.PrettyMIDI:
    """Return a deep-copied MIDI with melody (and optionally drums) removed."""
    midi_copy = copy.deepcopy(midi)
    if keep_melody:
        if include_drums:
            return midi_copy
        midi_copy.instruments = [instrument for instrument in midi_copy.instruments if not instrument.is_drum]
        return midi_copy

    melody_name_set = {name.lower() for name in melody_names if name}
    melody_program_set = set(melody_programs)
    melody_index_set = set(melody_indices)

    filtered: List[pretty_midi.Instrument] = []
    for idx, instrument in enumerate(midi_copy.instruments):
        if not include_drums and instrument.is_drum:
            continue
        name = (instrument.name or "").strip().lower()
        is_melody = False
        if melody_name_set and name and name in melody_name_set:
            is_melody = True
        if melody_program_set and instrument.program in melody_program_set:
            is_melody = True
        if melody_index_set and idx in melody_index_set:
            is_melody = True
        if not is_melody:
            filtered.append(instrument)
    midi_copy.instruments = filtered
    return midi_copy


def _build_ground_truth_accompaniment_midi(
    midi: pretty_midi.PrettyMIDI,
    include_drums: bool,
) -> pretty_midi.PrettyMIDI:
    """Return a deep-copied MIDI keeping only the accompaniment instruments."""
    midi_copy = copy.deepcopy(midi)
    selected, _ = _select_ground_truth_instruments(midi_copy, include_drums)
    midi_copy.instruments = selected
    return midi_copy


def _is_accompaniment_instrument(instrument: pretty_midi.Instrument) -> bool:
    name = (instrument.name or "").strip().lower()
    return name in _ACCOMPANIMENT_TRACK_NAMES


def _select_ground_truth_instruments(
    midi: pretty_midi.PrettyMIDI,
    include_drums: bool,
) -> Tuple[List[pretty_midi.Instrument], bool]:
    available = [inst for inst in midi.instruments if include_drums or not inst.is_drum]
    piano_tracks = [inst for inst in available if _is_accompaniment_instrument(inst)]
    if piano_tracks:
        return piano_tracks, True
    return available, False


def _collect_ground_truth_accompaniment_notes(
    midi: pretty_midi.PrettyMIDI,
    include_drums: bool,
    piece_label: Optional[str] = None,
) -> List[pretty_midi.Note]:
    instruments, used_named_track = _select_ground_truth_instruments(midi, include_drums)
    if not instruments:
        if piece_label:
            LOGGER.warning("No usable instruments found in ground truth %s", piece_label)
        return []
    if not used_named_track and piece_label:
        LOGGER.warning(
            "Ground truth %s is missing a Piano track; using %d fallback instrument(s).",
            piece_label,
            len(instruments),
        )
    notes: List[pretty_midi.Note] = []
    for instrument in instruments:
        notes.extend(instrument.notes)
    return notes


_PROMPT_DIR_PATTERN = re.compile(r"prompt(?:len)?[_\-]?(\d+)", re.IGNORECASE)


def _infer_prompt_accompaniment_path(
    generated_path: Path,
    groundtruth_path: Path,
) -> Optional[Path]:
    """Return the prompt accompaniment path inferred from directory naming conventions."""
    for parent in groundtruth_path.parents:
        match = _PROMPT_DIR_PATTERN.match(parent.name)
        if not match:
            # print(f"Debug: Directory name {parent.name} does not match prompt pattern.")
            continue
        prompt_len = match.group(1)
        name_variants: set[str] = {groundtruth_path.name}
        stem = groundtruth_path.stem
        if stem:
            name_variants.add(stem)
        trimmed = stem
        while trimmed and trimmed.endswith(".mid"):
            trimmed = trimmed[:-4]
            if trimmed:
                name_variants.add(trimmed)
        candidate_names = [f"{variant}_promptlen{prompt_len}.mid" for variant in name_variants]

        search_roots: set[Path] = set()
        current = parent
        for _ in range(4):
            search_roots.add(current)
            if current == current.parent:
                break
            current = current.parent

        for root in search_roots:
            for candidate_name in candidate_names:
                candidate = root / candidate_name
                if candidate.exists():
                    return candidate
    return None


class _FrechetDatasetBuilder:
    """Prepare accompaniment-only datasets for Frechet Music Distance computation."""

    def __init__(self, root: Path, args: argparse.Namespace) -> None:
        self.root = root
        self.generated_dir = root / "generated"
        self.groundtruth_dir = root / "groundtruth"
        self.generated_dir.mkdir(parents=True, exist_ok=True)
        self.groundtruth_dir.mkdir(parents=True, exist_ok=True)
        self._args = args
        self._stems: set[str] = set()

    def add_pair(
        self,
        generated_path: Path,
        generated_midi: pretty_midi.PrettyMIDI,
        groundtruth_path: Path,
        ground_truth_midi: pretty_midi.PrettyMIDI,
    ) -> tuple[Path, Path]:
        stem = generated_path.stem
        if stem in self._stems:
            raise ValueError(f"Duplicate stem encountered while preparing FMD dataset: {stem}")

        generated_accompaniment = _build_accompaniment_only_midi(
            generated_midi,
            self._args.melody_track_names,
            self._args.melody_programs,
            self._args.melody_track_indices,
            self._args.keep_melody,
            self._args.include_drums,
        )
        generated_dest = self.generated_dir / f"{stem}.mid"
        generated_accompaniment.write(str(generated_dest))

        groundtruth_accompaniment = _build_ground_truth_accompaniment_midi(
            ground_truth_midi,
            self._args.include_drums,
        )
        groundtruth_dest = self.groundtruth_dir / f"{stem}.mid"
        groundtruth_accompaniment.write(str(groundtruth_dest))

        self._stems.add(stem)
        return generated_dest, groundtruth_dest

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


def _compute_frechet_music_distance_dataset(
    reference_dir: Path,
    test_dir: Path,
    model_name: str,
    estimator_name: str,
    verbose: bool,
    cache_dir: Optional[Path],
) -> Optional[float]:
    """Compute Frechet Music Distance between two prepared accompaniment datasets."""
    if not reference_dir.exists() or not test_dir.exists():
        LOGGER.warning("FMD dataset directories missing; skipping FMD computation.")
        return None
    if not any(reference_dir.iterdir()) or not any(test_dir.iterdir()):
        LOGGER.warning("FMD dataset directories are empty; skipping FMD computation.")
        return None

    home_override = Path.cwd()
    original_home = os.environ.get("HOME")
    os.environ["HOME"] = str(home_override)
    try:
        try:
            from frechet_music_distance import FrechetMusicDistance  # type: ignore
            from frechet_music_distance import memory as fmd_memory  # type: ignore
        except ImportError:
            LOGGER.error(
                "frechet-music-distance package not available. Install it with 'pip install frechet-music-distance'."
            )
            return None
    finally:
        if original_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = original_home

    target_cache = cache_dir or (home_override / ".cache" / "frechet_music_distance" / "precomputed")
    target_cache = target_cache.resolve()
    target_cache.mkdir(parents=True, exist_ok=True)

    try:
        from joblib import Memory

        fmd_memory.MEMORY = Memory(str(target_cache), verbose=0)
        metric = FrechetMusicDistance(
            feature_extractor=model_name,
            gaussian_estimator=estimator_name,
            verbose=verbose,
        )
        score = metric.score(str(reference_dir), str(test_dir))
    except Exception as exc:  # pragma: no cover - depends on external package/runtime
        LOGGER.error("Failed to compute Frechet Music Distance: %s", exc)
        return None

    return float(score)


def _five_number_summary(values: Iterable[Optional[float]]) -> Optional[dict]:
    numeric_values: List[float] = []
    for val in values:
        if val is None:
            continue
        try:
            numeric = float(val)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        numeric_values.append(numeric)
    if not numeric_values:
        return None
    arr = np.asarray(numeric_values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "q1": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.5)),
        "q3": float(np.quantile(arr, 0.75)),
        "max": float(arr.max()),
        "count": int(arr.size),
    }


def _format_optional(value: Optional[float], precision: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(numeric):
        return "n/a"
    return f"{numeric:.{precision}f}"


def _is_stat_dict(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    required = {"mean", "min", "q1", "median", "q3", "max"}
    return required.issubset(value.keys())


def _print_summary_block(label: str, summary: Optional[object], fmt: callable) -> None:
    if summary is None:
        print(f"  {label}: n/a")
        return
    if _is_stat_dict(summary):
        print(f"  {label}:")
        count = summary.get("count")
        if count is not None:
            print(f"    count: {count}")
        print(f"    mean: {fmt(summary.get('mean'))}")
        print(f"    min: {fmt(summary.get('min'))}")
        print(f"    q1: {fmt(summary.get('q1'))}")
        print(f"    median: {fmt(summary.get('median'))}")
        print(f"    q3: {fmt(summary.get('q3'))}")
        print(f"    max: {fmt(summary.get('max'))}")
        return
    if isinstance(summary, dict):
        print(f"  {label}:")
        for key, value in summary.items():
            pretty = key.replace("_", " ")
            if _is_stat_dict(value):
                print(f"    {pretty}:")
                count = value.get("count")
                if count is not None:
                    print(f"      count: {count}")
                print(f"      mean: {fmt(value.get('mean'))}")
                print(f"      min: {fmt(value.get('min'))}")
                print(f"      q1: {fmt(value.get('q1'))}")
                print(f"      median: {fmt(value.get('median'))}")
                print(f"      q3: {fmt(value.get('q3'))}")
                print(f"      max: {fmt(value.get('max'))}")
            else:
                print(f"    {pretty}: {value if value is not None else 'n/a'}")
        return
    if isinstance(summary, (int, float)):
        print(f"  {label}: {fmt(summary)}")
        return
    print(f"  {label}: {summary}")


def evaluate_pair(
    generated_path: Path,
    groundtruth_path: Path,
    args: argparse.Namespace,
    frechet_builder: Optional[_FrechetDatasetBuilder] = None,
    prompt_path: Optional[Path] = None,
) -> dict:
    generated_midi = _load_midi(generated_path)
    ground_truth_midi = _load_midi(groundtruth_path)

    prompt_midi: Optional[pretty_midi.PrettyMIDI] = None
    prompt_notes: List[pretty_midi.Note] = []
    prompt_continuation_notes: List[pretty_midi.Note] = []
    generated_continuation_notes: List[pretty_midi.Note] = []
    prompt_end_time: Optional[float] = None
    if prompt_path is not None:
        try:
            prompt_midi = _load_midi(prompt_path)
        except Exception as exc:  # pragma: no cover - best effort logging
            LOGGER.warning("Failed to load prompt accompaniment for %s: %s", generated_path.name, exc)
        else:
            prompt_notes = _collect_accompaniment_notes(
                prompt_midi,
                args.melody_track_names,
                args.melody_programs,
                args.melody_track_indices,
                args.keep_melody,
                args.include_drums,
            )
            end_candidate = float(prompt_midi.get_end_time())
            if end_candidate > 0:
                prompt_end_time = end_candidate

    generated_notes = _collect_accompaniment_notes(
        generated_midi,
        args.melody_track_names,
        args.melody_programs,
        args.melody_track_indices,
        args.keep_melody,
        args.include_drums,
    )
    melody_notes = _collect_melody_notes(
        generated_midi,
        args.melody_track_names,
        args.melody_programs,
        args.melody_track_indices,
        args.include_drums,
    )
    ground_truth_notes = _collect_ground_truth_accompaniment_notes(
        ground_truth_midi,
        args.include_drums,
        groundtruth_path.stem,
    )

    if prompt_end_time is not None:
        if ground_truth_notes:
            for note in ground_truth_notes:
                if note.end <= prompt_end_time:
                    continue
                clipped_start = max(note.start, prompt_end_time)
                if clipped_start >= note.end:
                    continue
                prompt_continuation_notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=clipped_start,
                        end=note.end,
                    )
                )
        if generated_notes:
            for note in generated_notes:
                if note.end <= prompt_end_time:
                    continue
                clipped_start = max(note.start, prompt_end_time)
                if clipped_start >= note.end:
                    continue
                generated_continuation_notes.append(
                    pretty_midi.Note(
                        velocity=note.velocity,
                        pitch=note.pitch,
                        start=clipped_start,
                        end=note.end,
                    )
                )

    if not generated_notes:
        LOGGER.warning("No accompaniment notes extracted from generated file %s", generated_path.name)
    if not melody_notes:
        LOGGER.warning("No melody notes extracted from generated file %s", generated_path.name)
    if not ground_truth_notes:
        LOGGER.warning("No notes found in ground truth file %s", groundtruth_path.name)
    if prompt_path is not None and not prompt_notes:
        LOGGER.warning("No accompaniment notes extracted from prompt file %s", prompt_path.name)

    config, piece_length = _determine_distribution_bins(
        generated_notes,
        ground_truth_notes,
        args.onset_bins,
        args.duration_bins,
        args.duration_clip_quantile,
    )

    pitch_generated = _compute_pitch_histogram(generated_notes)
    pitch_ground_truth = _compute_pitch_histogram(ground_truth_notes)
    onset_generated = _compute_onset_histogram(generated_notes, piece_length, config.onset_bins)
    onset_ground_truth = _compute_onset_histogram(ground_truth_notes, piece_length, config.onset_bins)
    duration_generated = _compute_duration_histogram(generated_notes, config.duration_bins)
    duration_ground_truth = _compute_duration_histogram(ground_truth_notes, config.duration_bins)
    harmonicity = _compute_harmonic_consonance(melody_notes, generated_notes, args.consonant_intervals)
    chord_accuracy = _compute_chord_accuracy(
        generated_path.stem,
        generated_notes,
        args.chord_annotation_root,
    )
    prompt_polydis = None
    prompt_generated_continuation_polydis = None
    prompt_groundtruth_continuation_polydis = None
    if args.polydis_root is not None:
        if prompt_midi is None:
            if prompt_path is not None:
                LOGGER.warning("Skipping PolyDis metric for %s due to prompt load failure.", generated_path.name)
            else:
                LOGGER.warning("Prompt accompaniment missing for %s; skipping PolyDis metric.", generated_path.name)
        elif not prompt_notes:
            LOGGER.warning("Insufficient prompt notes for PolyDis metric in %s", generated_path.name)
        else:
            prompt_polydis = _compute_polydis_latent_similarity(
                generated_midi,
                generated_notes,
                prompt_midi,
                prompt_notes,
                args.polydis_root,
            )
            if prompt_end_time is not None:
                prompt_generated_continuation_polydis = _compute_polydis_latent_similarity(
                    generated_midi,
                    generated_continuation_notes,
                    prompt_midi,
                    prompt_notes,
                    args.polydis_root,
                )
                if not generated_continuation_notes:
                    LOGGER.warning(
                        "Generated continuation is empty past prompt for %s; treating as silence in PolyDis.",
                        generated_path.name,
                    )

            if prompt_end_time is not None:
                prompt_groundtruth_continuation_polydis = _compute_polydis_latent_similarity(
                    prompt_midi,
                    prompt_notes,
                    ground_truth_midi,
                    prompt_continuation_notes,
                    args.polydis_root,
                )
                if not prompt_continuation_notes:
                    LOGGER.warning(
                        "Ground-truth continuation is empty past prompt for %s; treating as silence in PolyDis.",
                        generated_path.name,
                    )

    auto_phrase_metrics = None
    if args.auto_phrase_analysis:
        generated_phrase_metrics = _segment_phrase_metrics_from_midi(
            generated_midi,
            generated_notes,
            generated_path.stem,
            args.phrase_bars,
            args.phrase_rhythm_resolution,
        )
        ground_phrase_metrics = _segment_phrase_metrics_from_midi(
            ground_truth_midi,
            ground_truth_notes,
            groundtruth_path.stem,
            args.phrase_bars,
            args.phrase_rhythm_resolution,
        )
        auto_phrase_metrics = {
            "generated": [asdict(metric) for metric in generated_phrase_metrics],
            "ground_truth": [asdict(metric) for metric in ground_phrase_metrics],
        }

    if frechet_builder is not None:
        try:
            frechet_builder.add_pair(
                generated_path,
                generated_midi,
                groundtruth_path,
                ground_truth_midi,
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            LOGGER.warning("Failed to prepare Frechet dataset for %s: %s", generated_path.name, exc)

    return {
        "piece": generated_path.stem,
        "generated_path": str(generated_path),
        "groundtruth_path": str(groundtruth_path),
        "prompt_path": str(prompt_path) if prompt_path is not None else None,
        "note_counts": {
            "generated": len(generated_notes),
            "ground_truth": len(ground_truth_notes),
        },
        "pitch_jsd": _js_divergence(pitch_generated, pitch_ground_truth),
        "onset_jsd": _js_divergence(onset_generated, onset_ground_truth),
        "duration_jsd": _js_divergence(duration_generated, duration_ground_truth),
        "harmonicity": harmonicity,
        "chord_accuracy": chord_accuracy,
        "prompt_polydis": prompt_polydis,
        "prompt_generated_continuation_polydis": prompt_generated_continuation_polydis,
        "prompt_groundtruth_continuation_polydis": prompt_groundtruth_continuation_polydis,
        "auto_phrase_metrics": auto_phrase_metrics,
        "histograms": {
            "pitch_generated": pitch_generated.tolist(),
            "pitch_ground_truth": pitch_ground_truth.tolist(),
            "onset_generated": onset_generated.tolist(),
            "onset_ground_truth": onset_ground_truth.tolist(),
            "duration_generated": duration_generated.tolist(),
            "duration_ground_truth": duration_ground_truth.tolist(),
            "onset_edges": config.onset_bins.tolist(),
            "duration_edges": config.duration_bins.tolist(),
        },
    }


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s | %(message)s")

    if not args.generated_dir.is_dir():
        parser.error(f"Generated directory not found: {args.generated_dir}")
    if not args.groundtruth_dir.is_dir():
        parser.error(f"Ground-truth directory not found: {args.groundtruth_dir}")
    if args.validation_phrases is not None and not args.validation_phrases.is_file():
        parser.error(f"Validation phrases file not found: {args.validation_phrases}")

    pairs = _match_ground_truth(args.generated_dir, args.groundtruth_dir)
    if not pairs:
        LOGGER.error("No matching MIDI pairs found.")
        return

    frechet_builder: Optional[_FrechetDatasetBuilder] = None
    frechet_temp_root: Optional[Path] = None
    if args.frechet_music_distance:
        try:
            frechet_temp_root = Path(tempfile.mkdtemp(prefix="fmd_", dir=str(Path.cwd())))
            frechet_builder = _FrechetDatasetBuilder(frechet_temp_root, args)
        except Exception as exc:  # pragma: no cover - filesystem/environment dependent
            LOGGER.error("Failed to prepare Frechet Music Distance workspace: %s", exc)
            frechet_builder = None
            if frechet_temp_root is not None:
                shutil.rmtree(frechet_temp_root, ignore_errors=True)

    results: List[dict] = []
    for gen_path, gt_path in pairs:
        prompt_path = _infer_prompt_accompaniment_path(gen_path, gt_path)
        results.append(
            evaluate_pair(
                gen_path,
                gt_path,
                args,
                frechet_builder=frechet_builder,
                prompt_path=prompt_path,
            )
        )

    pitch_jsd = _five_number_summary(result["pitch_jsd"] for result in results)
    onset_jsd = _five_number_summary(result["onset_jsd"] for result in results)
    duration_jsd = _five_number_summary(result["duration_jsd"] for result in results)
    harmonic_values = [result["harmonicity"] for result in results if result["harmonicity"]]
    harmonic_summary: Optional[dict] = None
    if harmonic_values:
        harmonic_summary = {}
        keys = harmonic_values[0].keys()
        for key in keys:
            harmonic_summary[key] = _five_number_summary(item.get(key) for item in harmonic_values)
    chord_accuracy_summary = _five_number_summary(result.get("chord_accuracy") for result in results)
    phrase_pair_summary = None
    if args.validation_phrases is not None:
        phrase_groups = _load_phrase_metrics(
            args.validation_phrases,
            args.phrase_window_seconds,
            args.phrase_rhythm_resolution,
        )
        phrase_pair_summary = _analyze_phrase_pair_consistency(phrase_groups, max(0, args.pair_analysis_random))
    auto_phrase_summary = None
    if args.auto_phrase_analysis:
        rd_diffs: List[float] = []
        vn_diffs: List[float] = []
        for result in results:
            auto_metrics = result.get("auto_phrase_metrics") or {}
            generated_segments = auto_metrics.get("generated", [])
            ground_segments = auto_metrics.get("ground_truth", [])
            total_pairs = min(len(generated_segments), len(ground_segments))
            for idx in range(total_pairs):
                gen = generated_segments[idx]
                gt = ground_segments[idx]
                rd_diffs.append(abs(float(gen["rhythm_density"]) - float(gt["rhythm_density"])))
                vn_diffs.append(abs(float(gen["voice_number"]) - float(gt["voice_number"])))
        if rd_diffs and vn_diffs:
            auto_phrase_summary = {
                "count": len(rd_diffs),
                "rd": _five_number_summary(rd_diffs),
                "vn": _five_number_summary(vn_diffs),
            }
    prompt_polydis_summary = None
    prompt_generated_continuation_polydis_summary = None
    prompt_groundtruth_continuation_polydis_summary = None
    if args.polydis_root is not None:
        prompt_polydis_values = [result.get("prompt_polydis") for result in results if result.get("prompt_polydis")]
        if prompt_polydis_values:
            prompt_polydis_summary = {
                "txt_mean_distance": _five_number_summary(
                    val.get("txt_mean_distance") for val in prompt_polydis_values
                ),
                "chd_mean_distance": _five_number_summary(
                    val.get("chd_mean_distance") for val in prompt_polydis_values
                ),
                "segments": _five_number_summary(val.get("segments") for val in prompt_polydis_values),
            }

        generated_values = [
            result.get("prompt_generated_continuation_polydis")
            for result in results
            if result.get("prompt_generated_continuation_polydis")
        ]
        if generated_values:
            prompt_generated_continuation_polydis_summary = {
                "txt_mean_distance": _five_number_summary(val.get("txt_mean_distance") for val in generated_values),
                "chd_mean_distance": _five_number_summary(val.get("chd_mean_distance") for val in generated_values),
                "segments": _five_number_summary(val.get("segments") for val in generated_values),
            }

        groundtruth_values = [
            result.get("prompt_groundtruth_continuation_polydis")
            for result in results
            if result.get("prompt_groundtruth_continuation_polydis")
        ]
        if groundtruth_values:
            prompt_groundtruth_continuation_polydis_summary = {
                "txt_mean_distance": _five_number_summary(val.get("txt_mean_distance") for val in groundtruth_values),
                "chd_mean_distance": _five_number_summary(val.get("chd_mean_distance") for val in groundtruth_values),
                "segments": _five_number_summary(val.get("segments") for val in groundtruth_values),
            }

    frechet_score = None
    if frechet_builder is not None:
        frechet_score = _compute_frechet_music_distance_dataset(
            frechet_builder.groundtruth_dir,
            frechet_builder.generated_dir,
            args.frechet_model,
            args.frechet_estimator,
            args.frechet_verbose,
            args.frechet_cache_dir,
        )
        if args.frechet_keep_temp:
            LOGGER.info(
                "Frechet Music Distance temporary data preserved at %s",
                frechet_builder.root,
            )
        else:
            frechet_builder.cleanup()
    elif args.frechet_music_distance:
        LOGGER.warning("Frechet Music Distance was requested but could not be prepared.")

    accompaniment_vs_gt_summary = {
        "pitch_jsd": pitch_jsd,
        "onset_jsd": onset_jsd,
        "duration_jsd": duration_jsd,
        "prompt_polydis": prompt_polydis_summary,
        "prompt_generated_continuation_polydis": prompt_generated_continuation_polydis_summary,
        "prompt_groundtruth_continuation_polydis": prompt_groundtruth_continuation_polydis_summary,
        "frechet_music_distance": frechet_score,
    }

    inter_track_continuity_summary = {
        "validation_pairs": phrase_pair_summary,
        "auto_phrase_pairs": auto_phrase_summary,
    }

    melody_relationship_summary = {
        "harmonicity": harmonic_summary,
        "chord_accuracy": chord_accuracy_summary,
    }

    fmt = _format_optional

    print(f"Pairs evaluated: {len(results)}")

    print("Accompaniment vs Ground Truth:")
    _print_summary_block("Pitch JSD", accompaniment_vs_gt_summary.get("pitch_jsd"), fmt)
    _print_summary_block("Onset JSD", accompaniment_vs_gt_summary.get("onset_jsd"), fmt)
    _print_summary_block("Duration JSD", accompaniment_vs_gt_summary.get("duration_jsd"), fmt)
    _print_summary_block("Frechet Music Distance", accompaniment_vs_gt_summary.get("frechet_music_distance"), fmt)
    _print_summary_block("PolyDis (prompt)", accompaniment_vs_gt_summary.get("prompt_polydis"), fmt)
    _print_summary_block(
        "PolyDis (prompt vs generated continuation)",
        accompaniment_vs_gt_summary.get("prompt_generated_continuation_polydis"),
        fmt,
    )
    _print_summary_block(
        "PolyDis (prompt vs ground truth continuation)",
        accompaniment_vs_gt_summary.get("prompt_groundtruth_continuation_polydis"),
        fmt,
    )

    print()
    print("Accompaniment Inter-Track Continuity:")
    validation_pairs = inter_track_continuity_summary.get("validation_pairs")
    if validation_pairs:
        for label, key in (("Random", "random"), ("Same Song", "same_song"), ("Adjacent", "adjacent")):
            _print_summary_block(f"Validation {label}", validation_pairs.get(key), fmt)
    else:
        print("  Validation phrase pairs: n/a")

    auto_pairs = inter_track_continuity_summary.get("auto_phrase_pairs")
    if auto_pairs:
        _print_summary_block("Auto phrase (gen vs gt)", auto_pairs, fmt)
    elif args.auto_phrase_analysis:
        print("  Auto phrase (gen vs gt): n/a")

    print()
    print("Accompaniment ↔ Melody:")
    _print_summary_block("Harmonicity", melody_relationship_summary.get("harmonicity"), fmt)
    _print_summary_block("Chord Accuracy", melody_relationship_summary.get("chord_accuracy"), fmt)

    if args.output_json:
        payload = {
            "meta": {
                "pairs": len(results),
            },
            "summary": {
                "accompaniment_vs_groundtruth": accompaniment_vs_gt_summary,
                "inter_track_continuity": inter_track_continuity_summary,
                "melody_relationship": melody_relationship_summary,
            },
            "details": results,
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"Saved detailed metrics to {args.output_json}")


if __name__ == "__main__":
    main()
