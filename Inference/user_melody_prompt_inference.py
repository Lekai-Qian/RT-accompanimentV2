from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pretty_midi
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import ModelConfig, TrainingConfig
from inference_new import load_model, resolve_device, save_json, seed_everything
from my_tokenizer import GenerationStep
from PianoDataset import PianoDataset
from Token2Midi import MidiConverter


PROMPT_MODES = ("with_prompt", "mel_only_prompt", "no_prompt")


def resolve_repo_path(path_str: str | None) -> str | None:
    if not path_str:
        return path_str
    expanded = os.path.expanduser(path_str)
    if os.path.isabs(expanded):
        return expanded
    return str(ROOT_DIR / expanded)


def parse_csv_ints(value: str) -> List[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    return seeds


def parse_piece_specs(specs: List[str]) -> List[Dict[str, str]]:
    pieces = []
    for spec in specs:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise ValueError(
                "Each --piece must use piece_id:label:melody_midi_path, "
                f"got: {spec}"
            )
        piece_id, label, melody_path = parts
        pieces.append(
            {
                "piece_id": piece_id.strip(),
                "label": label.strip(),
                "melody_midi_path": resolve_repo_path(melody_path.strip()),
            }
        )
    if not pieces:
        raise ValueError("At least one --piece is required")
    return pieces


def find_npz_by_piece_id(data_dir: str, piece_id: str) -> str:
    exact = Path(data_dir) / f"{piece_id}.npz"
    if exact.is_file():
        return str(exact)
    matches = sorted(Path(data_dir).glob(f"*{piece_id}*.npz"))
    if len(matches) == 1:
        return str(matches[0])
    if not matches:
        raise FileNotFoundError(f"No dataset npz found for piece_id={piece_id} under {data_dir}")
    raise RuntimeError(f"Ambiguous dataset npz for piece_id={piece_id}: {matches}")


def load_dataset_piece(data_dir: str, piece_id: str) -> Tuple[str, dict, List[np.ndarray]]:
    npz_path = find_npz_by_piece_id(data_dir, piece_id)
    save_dict = np.load(npz_path, allow_pickle=True)
    metadata = save_dict["metadata"].item()
    measures = [save_dict[f"measure_{i}"].copy() for i in range(metadata["num_measures"])]
    return npz_path, metadata, measures


def primary_tempo_bpm(midi: pretty_midi.PrettyMIDI) -> float:
    _, tempos = midi.get_tempo_changes()
    if len(tempos) == 0:
        return 120.0
    tempo = float(tempos[0])
    if tempo <= 0:
        return 120.0
    return tempo


def melody_midi_to_roll(
    melody_midi_path: str,
    total_steps: int,
    timesteps_per_beat: int = 4,
) -> Tuple[np.ndarray, Dict]:
    """
    Convert an external melody MIDI to the tokenizer pianoroll grid.

    The conversion uses the MIDI's own beat grid, not the target dataset BPM.
    This keeps "beat 8" aligned musically when the source recording tempo differs
    from the dataset metadata tempo.
    """
    midi = pretty_midi.PrettyMIDI(melody_midi_path)
    source_bpm = primary_tempo_bpm(midi)
    sec_per_step = 60.0 / source_bpm / timesteps_per_beat

    roll = np.zeros((2, 88, total_steps), dtype=np.uint8)
    notes_seen = 0
    notes_used = 0
    clipped_notes = 0

    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            notes_seen += 1
            pitch_idx = int(note.pitch) - 21
            if pitch_idx < 0 or pitch_idx >= 88:
                continue

            start_step = int(round(note.start / sec_per_step))
            end_step = int(round(note.end / sec_per_step))
            if end_step <= start_step:
                end_step = start_step + 1

            if start_step >= total_steps or end_step <= 0:
                clipped_notes += 1
                continue

            clipped_start = max(0, start_step)
            clipped_end = min(total_steps, end_step)
            roll[0, pitch_idx, clipped_start:clipped_end] = 1
            roll[1, pitch_idx, clipped_start] = 1
            notes_used += 1
            if clipped_start != start_step or clipped_end != end_step:
                clipped_notes += 1

    info = {
        "melody_midi_path": melody_midi_path,
        "source_bpm": source_bpm,
        "sec_per_step": sec_per_step,
        "notes_seen": notes_seen,
        "notes_used": notes_used,
        "clipped_notes": clipped_notes,
        "source_end_time_sec": float(midi.get_end_time()),
        "source_estimated_steps": int(round(midi.get_end_time() / sec_per_step)),
        "target_total_steps": int(total_steps),
    }
    return roll, info


def replace_melody_channels(
    dataset_measures: List[np.ndarray],
    melody_roll: np.ndarray,
) -> List[np.ndarray]:
    measures = []
    cursor = 0
    for measure in dataset_measures:
        width = int(measure.shape[2])
        new_measure = measure.copy()
        new_measure[:2] = 0
        new_measure[:2, :, :] = melody_roll[:, :, cursor : cursor + width]
        measures.append(new_measure)
        cursor += width
    return measures


def extract_beat_plans(prep: Dict) -> List[Dict]:
    schedule = prep["schedule"]
    vocab = prep["vocab"]
    beat_plans: List[Dict] = []
    pending_prefix: List[torch.Tensor] = []

    for step in schedule:
        if step.action != "inject" or step.data is None:
            continue
        data = step.data.cpu().tolist()
        if len(data) == 1 and data[0] in (vocab.bar_token_id, vocab.beat_marker):
            pending_prefix.append(step.data.clone())
        elif len(data) > 0 and data[-1] == vocab.track_marker_mel:
            beat_plans.append(
                {
                    "prefix_tokens": [tok.clone() for tok in pending_prefix],
                    "mel_tokens": step.data.clone(),
                }
            )
            pending_prefix = []
    return beat_plans


def build_prompt_schedule(prep: Dict, prompt_mode: str, prompt_beats: int) -> List[GenerationStep]:
    if prompt_mode not in PROMPT_MODES:
        raise ValueError(f"Unsupported prompt_mode={prompt_mode}")

    vocab = prep["vocab"]
    empty_acc = torch.tensor([vocab.empty_marker, vocab.track_marker_acc], dtype=torch.long)
    beat_plans = extract_beat_plans(prep)
    acc_gt = prep["acc_beats_gt"]

    effective_prompt_beats = 0 if prompt_mode == "no_prompt" else max(0, prompt_beats)
    schedule: List[GenerationStep] = []

    for beat_idx, beat in enumerate(beat_plans):
        for tok in beat["prefix_tokens"]:
            schedule.append(GenerationStep("inject", tok.clone()))

        if beat_idx < effective_prompt_beats:
            if prompt_mode == "with_prompt":
                schedule.append(GenerationStep("inject", acc_gt[beat_idx].clone()))
            elif prompt_mode == "mel_only_prompt":
                schedule.append(GenerationStep("inject", empty_acc.clone()))
            else:
                schedule.append(GenerationStep("generate_acc"))
        else:
            schedule.append(GenerationStep("generate_acc"))

        schedule.append(GenerationStep("inject", beat["mel_tokens"].clone()))

    return schedule


def prepare_user_melody_generation(
    dataset: PianoDataset,
    data_dir: str,
    piece: Dict[str, str],
    prompt_mode: str,
    prompt_beats: int,
) -> Dict:
    tokenizer = dataset.tokenizer
    npz_path, metadata, dataset_measures = load_dataset_piece(data_dir, piece["piece_id"])
    total_steps = int(sum(int(measure.shape[2]) for measure in dataset_measures))

    melody_roll, melody_info = melody_midi_to_roll(
        piece["melody_midi_path"],
        total_steps=total_steps,
        timesteps_per_beat=4,
    )
    user_measures = replace_melody_channels(dataset_measures, melody_roll)

    gen_data = tokenizer.build_generation_schedule(
        measures=user_measures,
        metadata=metadata,
        gt_prefix_beats=0,
    )

    prep = {
        "piece_id": piece["piece_id"],
        "label": piece["label"],
        "gt_path": npz_path,
        "initial_tokens": gen_data["initial_tokens"],
        "schedule": gen_data["schedule"],
        "vocab": tokenizer.vocab,
        "mel_beats": gen_data["mel_beats"],
        "acc_beats_gt": gen_data["acc_beats_gt"],
        "melody_info": melody_info,
        "metadata": {
            "time_signature_idx": int(metadata.get("time_signature_idx", 4)),
            "bpm": int(metadata.get("bpm", 120) or 120),
            "num_measures": int(metadata["num_measures"]),
            "total_steps": total_steps,
        },
    }
    prep["prompt_schedule"] = build_prompt_schedule(prep, prompt_mode=prompt_mode, prompt_beats=prompt_beats)
    return prep


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inference with external user melody and dataset accompaniment prompt variants."
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--use-fp16", action="store_true")

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument(
        "--piece",
        action="append",
        default=[],
        help="piece_id:label:melody_midi_path; may be repeated",
    )
    parser.add_argument("--prompt-mode", choices=list(PROMPT_MODES), required=True)
    parser.add_argument("--prompt-beats", type=int, default=8)
    parser.add_argument("--seeds", type=str, default="42,43,44,45,46")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--export-dataset-gt-midi", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--temperature", type=float, default=1.1)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)

    args = parser.parse_args()

    if args.prompt_beats < 0:
        raise ValueError("--prompt-beats must be >= 0")

    ckpt_path = resolve_repo_path(args.ckpt)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    pieces = parse_piece_specs(args.piece)
    for piece in pieces:
        if not os.path.isfile(piece["melody_midi_path"]):
            raise FileNotFoundError(f"Melody MIDI not found: {piece['melody_midi_path']}")

    seeds = parse_csv_ints(args.seeds)
    seed_everything(seeds[0])
    device = resolve_device(args.device)

    model_config = ModelConfig()
    train_config = TrainingConfig()
    data_dir = resolve_repo_path(args.data_dir or train_config.data_dir)

    dataset = PianoDataset(
        data_dir=data_dir,
        config=model_config,
        cache_lengths=False,
        mode="test",
        test_split_ratio=0.10,
        random_seed=42,
    )

    model = load_model(
        model_path=ckpt_path,
        model_config=model_config,
        device=device,
        use_fp16=args.use_fp16,
    )
    converter = MidiConverter(dataset.tokenizer)

    output_dir = Path(resolve_repo_path(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    prepared = [
        prepare_user_melody_generation(
            dataset=dataset,
            data_dir=data_dir,
            piece=piece,
            prompt_mode=args.prompt_mode,
            prompt_beats=args.prompt_beats,
        )
        for piece in pieces
    ]

    manifest_items = []
    for seed in seeds:
        seed_everything(seed)
        for piece_idx, prep in enumerate(prepared):
            generator = (
                torch.Generator(device=device)
                if device.startswith("cuda") and torch.cuda.is_available()
                else torch.Generator()
            )
            generator.manual_seed(int(seed))

            acc_beats, mel_beats, generated_seq = model.generate_by_schedule(
                initial_tokens=prep["initial_tokens"],
                schedule=prep["prompt_schedule"],
                vocab=prep["vocab"],
                device=device,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                verbose=False,
                generator=generator,
            )

            mode_slug = {
                "with_prompt": f"withprompt{args.prompt_beats}",
                "mel_only_prompt": f"melprompt{args.prompt_beats}",
                "no_prompt": "noprompt0",
            }[args.prompt_mode]
            filename = (
                f"{prep['piece_id']}_{prep['label']}_{args.model_name}_"
                f"{mode_slug}_seed{seed}.mid"
            )
            gen_midi_path = output_dir / filename
            converter.beats_to_midi(
                mel_beats=mel_beats,
                acc_beats=acc_beats,
                tempo=prep["metadata"]["bpm"] or 120,
                save_path=str(gen_midi_path),
            )

            gt_midi_path = None
            if args.export_dataset_gt_midi and seed == seeds[0]:
                gt_midi_path = output_dir / f"{prep['piece_id']}_{prep['label']}_dataset_GT.mid"
                converter.gt_to_midi(prep["gt_path"], str(gt_midi_path))

            effective_prompt_beats = 0 if args.prompt_mode == "no_prompt" else int(args.prompt_beats)
            manifest_items.append(
                {
                    "piece_id": prep["piece_id"],
                    "label": prep["label"],
                    "seed": int(seed),
                    "model_name": args.model_name,
                    "prompt_mode": args.prompt_mode,
                    "prompt_beats": effective_prompt_beats,
                    "gt_path": prep["gt_path"],
                    "gen_midi_path": str(gen_midi_path),
                    "dataset_gt_midi_path": str(gt_midi_path) if gt_midi_path else None,
                    "metadata": prep["metadata"],
                    "melody_info": prep["melody_info"],
                    "generated_tokens": int(generated_seq.numel()),
                    "num_acc_beats": int(len(acc_beats)),
                    "num_mel_beats": int(len(mel_beats)),
                    "prompt_semantics": {
                        "melody_source": "external_user_melody_midi_all_beats",
                        "acc_source_before_prompt_end": (
                            "dataset_gt_acc"
                            if args.prompt_mode == "with_prompt"
                            else "empty_acc"
                            if args.prompt_mode == "mel_only_prompt"
                            else "model_generated_from_beat_0"
                        ),
                        "acc_source_after_prompt_end": "model_generated",
                    },
                }
            )
            print(
                f"done | model={args.model_name} | mode={args.prompt_mode} | "
                f"piece={prep['piece_id']} | seed={seed} | out={gen_midi_path.name}",
                flush=True,
            )

    fingerprint_input = "\n".join(
        f"{x['piece_id']}|{x['model_name']}|{x['prompt_mode']}|{x['seed']}"
        for x in manifest_items
    )
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_timestamp": run_timestamp,
        "device": device,
        "ckpt": ckpt_path,
        "model_name": args.model_name,
        "prompt_mode": args.prompt_mode,
        "prompt_beats": 0 if args.prompt_mode == "no_prompt" else int(args.prompt_beats),
        "seeds": seeds,
        "data_dir": data_dir,
        "generation": {
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "top_p": float(args.top_p),
            "repetition_penalty": float(args.repetition_penalty),
            "use_fp16": bool(args.use_fp16),
        },
        "selection_fingerprint_sha1": hashlib.sha1(fingerprint_input.encode("utf-8")).hexdigest(),
        "items": manifest_items,
    }
    save_json(output_dir / f"{run_timestamp}_manifest.json", manifest)
    print(f"complete | output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
