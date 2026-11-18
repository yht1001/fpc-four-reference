#!/usr/bin/env python3
"""Build support-only cycle_02/cycle_03 sheets for FPC threshold calibration."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from PIL import Image


OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
DATASET = ROOT / "datasets" / "FPCdataset2601"
FRAMES = DATASET / "frames" / "ffpmeg384fps25"
ANNOTATIONS = DATASET / "annotation" / "json" / "annotations.json"
SUPPORT = OUT / "artifacts" / "support_manifest.json"
DESTINATION = OUT / "threshold_calibration_v1"
FPS = 25.0
PANEL_SIZE = (384, 384)
MAX_CYCLE_SECONDS = 20.0


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def complete_cycles(metadata: dict) -> list[dict]:
    annotations = sorted(metadata["annotations"], key=lambda item: (item["segment"][0], item["segment"][1]))
    cycles = []
    for start_index, first in enumerate(annotations):
        if int(first["label_id"]) != 0:
            continue
        selected = [first]
        cursor = start_index + 1
        last_start = float(first["segment"][0])
        for wanted in range(1, 5):
            match = None
            while cursor < len(annotations):
                candidate = annotations[cursor]
                cursor += 1
                candidate_start = float(candidate["segment"][0])
                if candidate_start - float(first["segment"][0]) > MAX_CYCLE_SECONDS:
                    break
                if int(candidate["label_id"]) == wanted and candidate_start >= last_start:
                    match = candidate
                    last_start = candidate_start
                    break
            if match is None:
                selected = []
                break
            selected.append(match)
        if len(selected) != 5:
            continue
        if any(
            min(float(item["segment"][1]), float(other["segment"][1]))
            - max(float(item["segment"][0]), float(other["segment"][0])) > 0.05
            for item in selected
            for other in annotations
            if other is not item
        ):
            continue
        start = float(selected[0]["segment"][0])
        end = float(selected[-1]["segment"][1])
        if end - start <= MAX_CYCLE_SECONDS:
            cycles.append({
                "start_s": start,
                "end_s": end,
                "duration_s": round(end - start, 6),
                "events": [
                    {
                        "label_id": int(item["label_id"]),
                        "start_s": float(item["segment"][0]),
                        "end_s": float(item["segment"][1]),
                    }
                    for item in selected
                ],
            })
    return sorted(cycles, key=lambda item: item["start_s"])


def grid_floor(value: float) -> float:
    return round(math.floor((value - 0.25) / 0.5) * 0.5 + 0.25, 6)


def grid_ceil(value: float) -> float:
    return round(math.ceil((value - 0.25) / 0.5) * 0.5 + 0.25, 6)


def frame_number(timestamp_s: float, frame_count: int) -> int:
    zero_based = min(int(math.ceil(timestamp_s * FPS - 1e-9)), frame_count - 1)
    return max(0, zero_based) + 1


def build_split(source_video_id: str, role: str, start_center: float, end_center: float, frame_count: int) -> dict:
    pseudo_video_id = f"{source_video_id}__{role}"
    destination = DESTINATION / "sheets" / role
    destination.mkdir(parents=True, exist_ok=True)
    windows = []
    center = start_center
    window_id = 0
    while center <= end_center + 1e-8:
        end_s = center + 0.25
        requested = (end_s - 2.0, end_s - 1.0, end_s - 0.4, end_s - 0.2, end_s - 1 / FPS)
        images = []
        samples = []
        for sample_id, timestamp_s in enumerate(requested):
            number = frame_number(timestamp_s, frame_count)
            source = FRAMES / source_video_id / f"{number:06d}.jpg"
            image = Image.open(source).convert("RGB")
            if image.size != PANEL_SIZE:
                image = image.resize(PANEL_SIZE, Image.Resampling.LANCZOS)
            images.append(image)
            samples.append({
                "role": "context_1fps" if sample_id < 2 else "recent_5fps",
                "requested_s": round(timestamp_s, 6),
                "frame_index_1based": number,
                "source_path": str(source.resolve()),
            })
        sheet = Image.new("RGB", (5 * PANEL_SIZE[0] + 8, PANEL_SIZE[1]), "white")
        for index, image in enumerate(images):
            sheet.paste(image, (index * PANEL_SIZE[0] + (8 if index >= 2 else 0), 0))
            image.close()
        sheet_path = destination / f"window_{window_id:03d}.jpg"
        sheet.save(sheet_path, quality=90, subsampling=0)
        sheet.close()
        windows.append({
            "video_id": pseudo_video_id,
            "source_video_id": source_video_id,
            "split_role": role,
            "window_id": window_id,
            "target_start_s": round(center - 0.25, 6),
            "target_end_s": round(center + 0.25, 6),
            "target_center_s": round(center, 6),
            "samples": samples,
            "sheet": str(sheet_path.resolve()),
            "sheet_sha256": digest(sheet_path),
        })
        center = round(center + 0.5, 6)
        window_id += 1
    return {
        "video_id": pseudo_video_id,
        "source_video_id": source_video_id,
        "split_role": role,
        "fps": FPS,
        "windows": windows,
    }


def main() -> None:
    support = json.loads(SUPPORT.read_text(encoding="utf-8"))
    source_video_id = support["selected_cycle"]["video_id"]
    reference_start = float(support["selected_cycle"]["cycle_start_s"])
    database = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["database"]
    metadata = database[source_video_id]
    cycles = complete_cycles(metadata)
    reference_index = min(range(len(cycles)), key=lambda index: abs(cycles[index]["start_s"] - reference_start))
    if reference_index + 2 >= len(cycles):
        raise RuntimeError("selected reference cycle does not have two later complete cycles")
    calibration_cycle = cycles[reference_index + 1]
    validation_cycle = cycles[reference_index + 2]

    calibration_start = grid_floor(calibration_cycle["start_s"]) - 0.5
    calibration_end = grid_floor(validation_cycle["start_s"])
    validation_start = grid_ceil(validation_cycle["start_s"])
    validation_end = grid_floor(validation_cycle["end_s"] + 1.75)
    frame_count = int(metadata["duration_frame"])
    videos = [
        build_split(source_video_id, "calibration", calibration_start, calibration_end, frame_count),
        build_split(source_video_id, "validation", validation_start, validation_end, frame_count),
    ]
    manifest = {
        "schema_version": "jev-fpc-threshold-calibration-manifest-v1",
        "status": "frozen_for_inference",
        "source_support_video": source_video_id,
        "reference_cycle": {
            "start_s": float(support["selected_cycle"]["cycle_start_s"]),
            "end_s": float(support["selected_cycle"]["cycle_end_s"]),
        },
        "calibration_cycle": calibration_cycle,
        "validation_cycle": validation_cycle,
        "selection_policy": "the first two complete non-overlapping cycles after the frozen reference cycle in the same support video",
        "calibration_background_policy": "include the pre-cycle and inter-cycle interval through the final 0.5-second grid center before validation",
        "validation_background_policy": "include 1.75 seconds after the validation cycle",
        "ground_truth_used_in_prompt": False,
        "query_test_gt_used_for_selection": False,
        "sampling": {
            "stride_s": 0.5,
            "context": "preceding 2 s sampled at 1 fps (2 panels)",
            "target": "latest 0.5 s sampled at approximately 5 fps (3 panels)",
            "layout": "context, context | target, target, target",
        },
        "videos": videos,
        "total_windows": sum(len(video["windows"]) for video in videos),
    }
    DESTINATION.mkdir(parents=True, exist_ok=True)
    path = DESTINATION / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(path.resolve()),
        "source_support_video": source_video_id,
        "reference_cycle": manifest["reference_cycle"],
        "calibration_cycle": calibration_cycle,
        "validation_cycle": validation_cycle,
        "windows": {video["split_role"]: len(video["windows"]) for video in videos},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
