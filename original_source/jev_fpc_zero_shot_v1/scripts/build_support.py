#!/usr/bin/env python3
"""Select and materialise a deterministic cross-video Main-FPC support episode."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean

import cv2
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "FPCdataset2601"
OUT = ROOT / "jev_fpc_zero_shot_v1"
FRAMES = DATASET / "frames" / "ffpmeg384fps25"
ANNOTATIONS = DATASET / "annotation" / "json" / "annotations.json"
FRAME_GT = DATASET / "annotation" / "gtTXT"
TEST_SPLIT = DATASET / "split" / "nfold" / "split1.txt"
FRACTIONS = (0.20, 0.50, 0.80)
ROLES = ("early", "middle", "late")
MAX_CYCLE_SECONDS = 20.0

OPTION_MAP = {
    0: ("A01", "取手机将其放到工作台"),
    1: ("A02", "将FPC贴在挤压板上"),
    2: ("A03", "使用挤压板粘贴FPC"),
    3: ("A04", "FPC卡扣扣合"),
    4: ("A05", "手机放回传送带上"),
}

SUMMARY_DRAFTS = {
    "A01": "The operator picks up a phone and places it onto the central work surface for FPC assembly.",
    "A02": "The operator positions the flexible printed circuit onto the pressing plate and aligns it before bonding.",
    "A03": "The operator uses the pressing plate to press and bond the aligned flexible printed circuit onto the phone assembly.",
    "A04": "The operator aligns and presses the FPC connector or retaining buckle until the connection is visibly seated.",
    "A05": "After assembly, the operator moves the phone away from the work surface and returns it to the conveyor.",
    "BG": "None of the five defined FPC actions is visibly occurring; the interval is idle, transitional, or lacks a defining action cue.",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def frame_path(video_id: str, timestamp: float, fps: float, frame_count: int) -> tuple[Path, int]:
    index = max(1, min(frame_count, int(round(timestamp * fps)) + 1))
    path = FRAMES / video_id / f"{index:06d}.jpg"
    if not path.exists():
        raise FileNotFoundError(path)
    return path, index


def image_quality(path: Path) -> dict[str, float]:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot decode {path}")
    sharpness = float(cv2.Laplacian(image, cv2.CV_64F).var())
    brightness = float(image.mean())
    exposure = max(0.0, 1.0 - abs(brightness - 127.5) / 127.5)
    score = math.log1p(sharpness) + 0.75 * exposure
    return {
        "sharpness": round(sharpness, 4),
        "brightness": round(brightness, 4),
        "exposure_quality": round(exposure, 6),
        "quality_score": round(score, 6),
    }


def sample_segment(video_id: str, annotation: dict, fps: float, frame_count: int) -> dict:
    start, end = map(float, annotation["segment"])
    duration = end - start
    samples = []
    for role, fraction in zip(ROLES, FRACTIONS):
        timestamp = start + fraction * duration
        path, index = frame_path(video_id, timestamp, fps, frame_count)
        samples.append({
            "role": role,
            "fraction": fraction,
            "timestamp_s": round(timestamp, 6),
            "frame_index_1based": index,
            "source_path": str(path.resolve()),
            **image_quality(path),
        })
    qualities = [item["quality_score"] for item in samples]
    duration_bonus = min(max(duration, 0.0), 2.0) * 0.20
    score = mean(qualities) + 0.25 * min(qualities) + duration_bonus
    return {
        "label_id": int(annotation["label_id"]),
        "label": annotation["label"],
        "start_s": start,
        "end_s": end,
        "duration_s": round(duration, 6),
        "selection_score": round(score, 6),
        "samples": samples,
    }


def cycles_for_video(video_id: str, metadata: dict) -> list[dict]:
    annotations = sorted(metadata["annotations"], key=lambda item: (item["segment"][0], item["segment"][1]))
    fps = float(metadata["fps"])
    frame_count = int(metadata["duration_frame"])
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
        # A support card must depict one action at a time. The source annotations can
        # contain overlapping cycles (for example, returning one phone while picking up
        # the next). Exclude any cycle whose selected action overlaps another labelled
        # action by more than 50 ms.
        conflict = False
        for item in selected:
            start, end = map(float, item["segment"])
            for other in annotations:
                if other is item:
                    continue
                other_start, other_end = map(float, other["segment"])
                if min(end, other_end) - max(start, other_start) > 0.05:
                    conflict = True
                    break
            if conflict:
                break
        if conflict:
            continue
        sampled = [sample_segment(video_id, item, fps, frame_count) for item in selected]
        cycle_start = float(sampled[0]["start_s"])
        cycle_end = float(sampled[-1]["end_s"])
        if cycle_end - cycle_start > MAX_CYCLE_SECONDS:
            continue
        action_scores = [item["selection_score"] for item in sampled]
        cycle_score = mean(action_scores) + 0.40 * min(action_scores)
        cycles.append({
            "video_id": video_id,
            "cycle_start_s": cycle_start,
            "cycle_end_s": cycle_end,
            "cycle_duration_s": round(cycle_end - cycle_start, 6),
            "cycle_selection_score": round(cycle_score, 6),
            "actions": sampled,
        })
    return cycles


def background_runs(video_id: str, fps: float) -> list[dict]:
    labels = [line.strip() for line in (FRAME_GT / f"{video_id}.txt").read_text(encoding="utf-8").splitlines()]
    runs = []
    start = None
    for index, label in enumerate(labels + ["__END__"], start=1):
        if label == "background" and start is None:
            start = index
        elif label != "background" and start is not None:
            end = index - 1
            if end - start + 1 >= int(round(fps * 1.5)):
                midpoint = (start + end) // 2
                path = FRAMES / video_id / f"{midpoint:06d}.jpg"
                runs.append({
                    "start_frame_1based": start,
                    "end_frame_1based": end,
                    "start_s": round((start - 1) / fps, 6),
                    "end_s": round(end / fps, 6),
                    "duration_s": round((end - start + 1) / fps, 6),
                    "frame_index_1based": midpoint,
                    "timestamp_s": round((midpoint - 1) / fps, 6),
                    "source_path": str(path.resolve()),
                    **image_quality(path),
                })
            start = None
    for run in runs:
        run["selection_score"] = round(run["quality_score"] + 0.12 * min(run["duration_s"], 8.0), 6)
    return sorted(runs, key=lambda item: (-item["selection_score"], item["start_frame_1based"]))


def choose_background(runs: list[dict], count: int = 3) -> list[dict]:
    chosen = []
    for run in runs:
        if all(abs(run["timestamp_s"] - old["timestamp_s"]) >= 5.0 for old in chosen):
            chosen.append(run)
            if len(chosen) == count:
                return chosen
    for run in runs:
        if run not in chosen:
            chosen.append(run)
            if len(chosen) == count:
                return chosen
    raise RuntimeError(f"only found {len(chosen)} usable background references")


def make_sheet(paths: list[Path], destination: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in paths]
    width, height = images[0].size
    sheet = Image.new("RGB", (width * len(images), height))
    for index, image in enumerate(images):
        if image.size != (width, height):
            image = image.resize((width, height))
        sheet.paste(image, (index * width, 0))
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=95)
    for image in images:
        image.close()


def make_review_sheet(rows: list[tuple[str, Path, str]], destination: Path) -> None:
    rendered = []
    for option_id, path, subtitle in rows:
        image = Image.open(path).convert("RGB")
        header = Image.new("RGB", (image.width, 42), "white")
        draw = ImageDraw.Draw(header)
        draw.text((8, 5), f"{option_id}  {subtitle}", fill="black")
        panel = Image.new("RGB", (image.width, image.height + header.height), "white")
        panel.paste(header, (0, 0))
        panel.paste(image, (0, header.height))
        rendered.append(panel)
        image.close()
    review = Image.new("RGB", (rendered[0].width, sum(row.height for row in rendered)), "white")
    y = 0
    for row in rendered:
        review.paste(row, (0, y))
        y += row.height
        row.close()
    destination.parent.mkdir(parents=True, exist_ok=True)
    review.save(destination, quality=92)


def main() -> None:
    database = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))["database"]
    query_videos = sorted(line.strip() for line in TEST_SPLIT.read_text(encoding="utf-8").splitlines() if line.strip())
    support_pool = sorted(set(database) - set(query_videos))
    if len(query_videos) != 6 or len(support_pool) != 21 or set(query_videos) & set(support_pool):
        raise RuntimeError("unexpected support/query split")

    cycles = []
    per_video_counts = Counter()
    for video_id in support_pool:
        candidates = cycles_for_video(video_id, database[video_id])
        cycles.extend(candidates)
        per_video_counts[video_id] = len(candidates)
    if not cycles:
        raise RuntimeError("no complete five-action cycles found")
    ranking = sorted(cycles, key=lambda item: (-item["cycle_selection_score"], item["video_id"], item["cycle_start_s"]))
    selected = ranking[0]
    selected_video = selected["video_id"]

    frame_root = OUT / "support" / "frames"
    reference_root = OUT / "support" / "references"
    review_rows = []
    for action in selected["actions"]:
        option_id, label_name = OPTION_MAP[action["label_id"]]
        copied = []
        for sample in action["samples"]:
            source = Path(sample["source_path"])
            destination = frame_root / option_id / f"{sample['role']}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            sample["copied_path"] = str(destination)
            sample["sha256"] = sha256(destination)
            copied.append(destination)
        reference = reference_root / f"{option_id}.jpg"
        make_sheet(copied, reference)
        action["option_id"] = option_id
        action["canonical_label"] = label_name
        action["reference_sheet"] = str(reference)
        action["reference_sheet_sha256"] = sha256(reference)
        review_rows.append((option_id, reference, f"{action['start_s']:.2f}-{action['end_s']:.2f}s"))

    bg = choose_background(background_runs(selected_video, float(database[selected_video]["fps"])))
    bg_copied = []
    for index, sample in enumerate(bg, start=1):
        source = Path(sample["source_path"])
        destination = frame_root / "BG" / f"reference_{index}.jpg"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sample["copied_path"] = str(destination)
        sample["sha256"] = sha256(destination)
        bg_copied.append(destination)
    bg_reference = reference_root / "BG.jpg"
    make_sheet(bg_copied, bg_reference)
    review_rows.append(("BG", bg_reference, ", ".join(f"{item['timestamp_s']:.2f}s" for item in bg)))

    review_sheet = OUT / "support" / "review" / "support_contact_sheet.jpg"
    make_review_sheet(review_rows, review_sheet)

    split_manifest = {
        "schema_version": "jev-fpc-cross-video-split-v1",
        "dataset": str(DATASET),
        "split_policy": "nfold split1 is frozen query/test; nfold splits 2-4 are support candidates",
        "support_query_overlap": False,
        "support_pool_videos": support_pool,
        "query_test_videos": query_videos,
        "selected_support_video": selected_video,
    }
    config_root = OUT / "config"
    config_root.mkdir(parents=True, exist_ok=True)
    (config_root / "split_manifest.json").write_text(
        json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    cards = {
        "schema_version": "jev-fpc-action-card-drafts-v1",
        "status": "draft_auto_needs_human_review",
        "canonical_labels_exposed_to_model": False,
        "source_support_video": selected_video,
        "cards": {
            option_id: {
                "canonical_label_id": label_id,
                "canonical_label": label_name,
                "option_summary_en": SUMMARY_DRAFTS[option_id],
                "reference_sheet": str(reference_root / f"{option_id}.jpg"),
            }
            for label_id, (option_id, label_name) in OPTION_MAP.items()
        } | {
            "BG": {
                "canonical_label_id": 5,
                "canonical_label": "background",
                "option_summary_en": SUMMARY_DRAFTS["BG"],
                "reference_sheet": str(bg_reference),
            }
        },
    }
    (config_root / "action_card_drafts.json").write_text(
        json.dumps(cards, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    manifest = {
        "schema_version": "jev-fpc-support-manifest-v1",
        "status": "draft_auto_needs_human_review",
        "selection_policy": {
            "support_video_count": 1,
            "required_action_order": [0, 1, 2, 3, 4],
            "reference_fractions": list(FRACTIONS),
            "quality_terms": ["laplacian_sharpness", "exposure", "segment_duration", "weakest_action"],
            "background_policy": "three distinct >=1.5s GT background runs from the selected support video",
        },
        "selected_cycle": selected,
        "background_references": bg,
        "reference_sheets": {
            **{action["option_id"]: action["reference_sheet"] for action in selected["actions"]},
            "BG": str(bg_reference),
        },
        "review_sheet": str(review_sheet),
        "candidate_cycle_count": len(cycles),
        "candidate_cycle_count_by_video": dict(per_video_counts),
        "top_10_candidates": [
            {
                "rank": rank,
                "video_id": item["video_id"],
                "cycle_start_s": item["cycle_start_s"],
                "cycle_end_s": item["cycle_end_s"],
                "cycle_selection_score": item["cycle_selection_score"],
            }
            for rank, item in enumerate(ranking[:10], start=1)
        ],
    }
    artifact_root = OUT / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    (artifact_root / "support_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "selected_support_video": selected_video,
        "cycle": [selected["cycle_start_s"], selected["cycle_end_s"]],
        "cycle_selection_score": selected["cycle_selection_score"],
        "candidate_cycles": len(cycles),
        "reference_sheets": manifest["reference_sheets"],
        "review_sheet": str(review_sheet),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
