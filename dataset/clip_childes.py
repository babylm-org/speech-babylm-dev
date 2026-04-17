#!/usr/bin/env python3
import argparse
import csv
import os
from collections import defaultdict
from multiprocessing import Pool, get_start_method, shared_memory
from pathlib import Path

import numpy as np
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract metadata from TALKBank transcripts (.cha), including speaker roles and ages, "
        )
    )
    parser.add_argument(
        "--metadata_path",
        default="./output/childes_metadata.csv",
        help="Path to the metadata CSV file.",
    )
    parser.add_argument(
        "--media_dir",
        default="/Volumes/data/childes_media",
        help="Root directory for all media files.",
    )
    parser.add_argument(
        "--output_dir",
        default="/Volumes/data/childes_clipped",
        help="Path to the output directory for clipped audio files.",
    )
    return parser.parse_args()


def _init_shared(shm_name, shape, dtype, sr_, channels_):
    global SHM, ARR, SHAPE, DTYPE, SR, CHANNELS
    SHM = shared_memory.SharedMemory(name=shm_name)
    ARR = np.ndarray(shape, dtype=dtype, buffer=SHM.buf)
    SHAPE, DTYPE, SR, CHANNELS = shape, dtype, sr_, channels_


def _write_one(args):
    s, e, out = args
    i0 = int(SR * s)
    i1 = int(SR * e)
    data = ARR[i0:i1]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    sf.write(out, data, SR, subtype="PCM_16")
    print(f"Wrote: {out} (duration: {e - s:.2f}s, frames: {i1 - i0})")
    return


def parse_segments(
    metadata_path: Path, output_dir: Path, media_dir: Path
) -> dict[Path, list[tuple[float, float, str]]]:
    items = defaultdict(list)
    with open(metadata_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            out_path, wav_path, s, e = (
                output_dir / f"{row['utt_id']}.wav",
                media_dir / row["path"].replace("mp3", "wav"),
                float(row["start_sec"]),
                float(row["end_sec"]),
            )
            if out_path.exists():
                print(f"File already exists. Skip it.: {out_path}")
                continue
            items[wav_path].append((s, e, out_path))
    return items


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    segs_dict = parse_segments(args.metadata_path, args.output_dir, args.media_dir)

    for wav_path, segs in segs_dict.items():
        # Decode audio file once
        data, sr = sf.read(wav_path, dtype="int16", always_2d=False)
        if data.ndim == 2:
            channels = data.shape[1]
        else:
            channels = 1

        # English: "To shared memory (read-only intended)"
        shape = data.shape
        dtype = data.dtype
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        shm_arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        shm_arr[:] = data
        total_frames = shape[0]

        def clamp(t):
            return max(0.0, min(float(t), total_frames / sr))

        segs = [(clamp(s), clamp(e), out_path) for (s, e, out_path) in segs if e > s]

        # Write in parallel with spawn method
        if get_start_method(allow_none=True) != "spawn":
            try:
                import multiprocessing as mp

                mp.set_start_method("spawn", force=True)
            except RuntimeError:
                pass

        with Pool(
            processes=args.jobs,
            initializer=_init_shared,
            initargs=(shm.name, shape, dtype, sr, channels),
        ) as pool:
            for _ in pool.imap_unordered(_write_one, segs, chunksize=16):
                pass

        shm.close()
        shm.unlink()


if __name__ == "__main__":
    main()
