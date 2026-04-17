from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pretty_midi


_ACCOMPANIMENT_TRACK_NAMES = {"piano"}


@dataclass
class DistributionConfig:
    pitch_bins: Sequence[int]
    onset_bins: np.ndarray
    duration_bins: np.ndarray


def load_midi(path: Path) -> pretty_midi.PrettyMIDI:
    return pretty_midi.PrettyMIDI(str(path))


def is_accompaniment_instrument(instrument: pretty_midi.Instrument) -> bool:
    name = (instrument.name or "").strip().lower()
    return name in _ACCOMPANIMENT_TRACK_NAMES


def select_ground_truth_instruments(
    midi: pretty_midi.PrettyMIDI,
    include_drums: bool,
) -> Tuple[List[pretty_midi.Instrument], bool]:
    available = [inst for inst in midi.instruments if include_drums or not inst.is_drum]
    piano_tracks = [inst for inst in available if is_accompaniment_instrument(inst)]
    if piano_tracks:
        return piano_tracks, True
    return available, False


def collect_ground_truth_accompaniment_notes(
    midi: pretty_midi.PrettyMIDI,
    include_drums: bool,
) -> List[pretty_midi.Note]:
    instruments, _ = select_ground_truth_instruments(midi, include_drums)
    notes: List[pretty_midi.Note] = []
    for instrument in instruments:
        notes.extend(instrument.notes)
    return notes


def collect_accompaniment_notes(
    midi: pretty_midi.PrettyMIDI,
    melody_names: Iterable[str],
    melody_programs: Iterable[int],
    melody_indices: Iterable[int],
    keep_melody: bool,
    include_drums: bool,
) -> List[pretty_midi.Note]:
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


def compute_pitch_histogram(notes: Sequence[pretty_midi.Note]) -> np.ndarray:
    pitches = [note.pitch for note in notes]
    hist, _ = np.histogram(pitches, bins=np.arange(129), density=False)
    return hist.astype(np.float64)


def normalize(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0:
        return np.zeros_like(values, dtype=np.float64)
    return values / total


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-10
    mask = p > 0
    ratio = (p[mask] + eps) / (q[mask] + eps)
    return float(np.sum(p[mask] * np.log(ratio)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if p.size == 0 or q.size == 0:
        return float("nan")
    p_norm = normalize(p)
    q_norm = normalize(q)
    m = 0.5 * (p_norm + q_norm)
    return 0.5 * kl_divergence(p_norm, m) + 0.5 * kl_divergence(q_norm, m)


def compute_onset_histogram(
    notes: Sequence[pretty_midi.Note],
    piece_length: float,
    bin_edges: np.ndarray,
) -> np.ndarray:
    if piece_length <= 0:
        return np.zeros(bin_edges.size - 1, dtype=np.float64)
    starts = [max(0.0, min(1.0, note.start / piece_length)) for note in notes]
    hist, _ = np.histogram(starts, bins=bin_edges, density=False)
    return hist.astype(np.float64)


def determine_distribution_bins(
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


def compute_eval_acc_style_jsd(ref_path: str, gen_path: str) -> tuple[float, float]:
    generated_midi = load_midi(Path(gen_path))
    ground_truth_midi = load_midi(Path(ref_path))

    generated_notes = collect_accompaniment_notes(
        generated_midi,
        melody_names=("melody",),
        melody_programs=(),
        melody_indices=(),
        keep_melody=False,
        include_drums=False,
    )
    ground_truth_notes = collect_ground_truth_accompaniment_notes(
        ground_truth_midi,
        include_drums=False,
    )

    config, piece_length = determine_distribution_bins(
        generated_notes,
        ground_truth_notes,
        onset_bins=64,
        duration_bins=64,
        duration_clip_quantile=0.995,
    )

    pitch_generated = compute_pitch_histogram(generated_notes)
    pitch_ground_truth = compute_pitch_histogram(ground_truth_notes)
    onset_generated = compute_onset_histogram(generated_notes, piece_length, config.onset_bins)
    onset_ground_truth = compute_onset_histogram(ground_truth_notes, piece_length, config.onset_bins)

    pitch_jsd = js_divergence(pitch_generated, pitch_ground_truth)
    onset_jsd = js_divergence(onset_generated, onset_ground_truth)
    return pitch_jsd, onset_jsd
