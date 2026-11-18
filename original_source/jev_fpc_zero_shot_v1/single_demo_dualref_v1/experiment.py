"""Prepare, select, predict and score the single annotated Demo experiment."""

from collections import Counter
from itertools import groupby, islice
from pathlib import Path
import argparse
import datetime
import hashlib
import math
import re

import numpy as np

from setup_study import BANKS, BLOCKS, DATA, DEMO, FPC, ROOT, backend as b
import build_threshold_calibration as calibration
from build_support import make_sheet
import evaluate_ensemble_temporal_prior as ensemble
import evaluate_probability_alignment as alignment
import evaluate_soft_temporal_prior as temporal
from evaluate_query import GT_TO_OPTION
from score import metrics


def split_ids(index):
    path = DATA / f"split/nfold/split{index}.txt"
    return path.read_text().split()


def recording_id(video):
    return re.sub(r"(?:_part\d+(?:_480p)?|_\d{3}|_480p)$", "", video)


def label_index(line):
    return b.LABELS.index(GT_TO_OPTION[line.strip()])


def demo_labels():
    with (b.GT / f"{DEMO}.txt").open() as stream:
        labels = [label_index(line) for line in islice(stream, 1500)]
    if len(labels) != 1500:
        raise ValueError("The fixed Demo has fewer than 1,500 labeled frames")
    return labels


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def reference_banks(labels):
    runs = []
    start = 0
    for label, group in groupby(labels[:750]):
        stop = start + sum(1 for _ in group)
        runs.append((label, start, stop))
        start = stop
    annotations = [
        {"label_id": label, "segment": [start / 25, stop / 25]}
        for label, start, stop in runs
        if label != 5 and stop < 750
    ]
    cycles = calibration.complete_cycles({"annotations": annotations})
    selected = [cycles[0]] if cycles else []
    for cycle in cycles[1:]:
        if cycle["start_s"] >= selected[-1]["end_s"]:
            selected.append(cycle)
            break
    if len(selected) != 2:
        raise ValueError("The fixed reference interval does not contain two complete cycles")

    backgrounds = [(start, stop) for label, start, stop in runs if label == 5 and stop - start >= 15]
    sheets = {}
    provenance = {}
    for bank, cycle in zip(BANKS, selected):
        bg = min(backgrounds, key=lambda span: (abs(sum(span) / 50 - cycle["end_s"]), span))
        backgrounds.remove(bg)
        spans = [
            (round(event["start_s"] * 25), round(event["end_s"] * 25))
            for event in cycle["events"]
        ] + [bg]
        records = {}
        hashes = []
        for index, (label, (start, stop)) in enumerate(zip(b.LABELS, spans)):
            frames = [start + math.floor(fraction * (stop - start)) for fraction in (0.2, 0.5, 0.8)]
            assert len(set(frames)) == 3 and all(labels[frame] == index for frame in frames)
            paths = [b.FRAMES / DEMO / f"{frame + 1:06d}.jpg" for frame in frames]
            dest = b.REFS[bank] / f"{label}.jpg"
            make_sheet(paths, dest)
            hashes.append(b.sha(dest))
            records[label] = {
                "source_frames_zero_based": frames,
                "source_sha256": {str(path): b.sha(path) for path in paths},
                "span_frames": [start, stop],
                "sheet": str(dest),
                "sheet_sha256": hashes[-1],
            }
        sheets[bank] = hashes
        provenance[bank] = {"cycle": cycle, "labels": records}
    b.save(ROOT / "REFERENCES.json", sheets)
    b.save(ROOT / "reference_provenance.json", provenance)


def prepare():
    if (ROOT / "RUN_LOCK.json").exists():
        raise FileExistsError("The study inputs are already frozen")
    query_ids = set(split_ids(1))
    query_recordings = {recording_id(video) for video in query_ids}
    candidates = [
        video for index in (2, 3, 4) for video in split_ids(index)
        if recording_id(video) not in query_recordings
    ]
    chosen = min(candidates, key=lambda video: hashlib.sha256(f"fpc-single-demo-v1:{video}".encode()).hexdigest())
    assert chosen == DEMO and DEMO not in query_ids
    b.save(ROOT / "DEMO_LOCK.json", {
        "created_utc": now(), "video_id": chosen, "candidate_ids": sorted(candidates),
        "query_ids": sorted(query_ids), "label_frame_range_zero_based": [0, 1500],
        "selection_uses_only_split_ids": True,
    })
    labels = demo_labels()
    reference_banks(labels)
    calibration.DESTINATION = ROOT
    support_rows = []
    for block, (start, stop) in BLOCKS.items():
        rows = calibration.build_split(DEMO, block, start + 0.25, stop - 0.25, 1500)["windows"]
        for row in rows:
            row.update(video_id=DEMO, window_id=len(support_rows), block=block)
            assert all((start - 2) * 25 <= sample["frame_index_1based"] - 1 < stop * 25 for sample in row["samples"])
            support_rows.append(row)
        observed = {labels[int(row["target_center_s"] * 25)] for row in rows}
        if observed != set(range(6)):
            raise ValueError(f"The fixed {block} block does not cover all six classes: {observed}")
    manifest_path = FPC / "query/manifest.json"
    manifest = b.read(manifest_path)
    assert {video["video_id"] for video in manifest["videos"]} == query_ids and len(query_ids) == 6
    query_rows = [
        {key: window[key] for key in ("video_id", "window_id", "target_start_s", "target_end_s", "target_center_s", "sheet", "sheet_sha256")}
        for video in manifest["videos"]
        for window in video["windows"]
    ]
    assert len(query_rows) == 4786 and len(support_rows) == 52
    for name, rows in (("support", support_rows), ("query", query_rows)):
        for row in rows:
            row["request_sha256"] = {bank: b.signature(row, bank) for bank in BANKS}
        b.save(ROOT / "inputs" / f"{name}.json", {"windows": rows})
    sources = [
        *ROOT.glob("*.py"), ROOT / "PROTOCOL.md", ROOT / "DEMO_LOCK.json", ROOT / "REFERENCES.json",
        ROOT / "reference_provenance.json", *ROOT.glob("inputs/*.json"), *ROOT.glob("references/*/*.jpg"),
        FPC / "bin02_dualref_v1/infer.py", FPC / "bin02_dualref_v1/bin02_env.py",
        FPC / "bin02_dualref_v1/score.py", manifest_path,
        *(FPC / f"scripts/{name}.py" for name in (
            "build_support", "build_threshold_calibration", "run_query", "evaluate_query",
            "evaluate_probability_alignment", "evaluate_ensemble_temporal_prior",
            "evaluate_soft_temporal_prior", "evaluate_probability_ensemble"
        )),
        *(DATA / f"split/nfold/split{index}.txt" for index in range(1, 5)),
    ]
    previous_model = b.read(FPC / "bin02_dualref_v1/INPUT_LOCK.json")["pretrained_weight_sha256"]
    b.save(ROOT / "RUN_LOCK.json", {
        "created_utc": now(), "source_sha256": {str(path): b.sha(path) for path in sorted(sources)},
        "weights_sha256": previous_model, "demo_label_prefix_sha256": b.fingerprint(labels),
        "support_windows": len(support_rows), "query_windows": len(query_rows),
    })


def fit():
    run_lock = b.read(ROOT / "RUN_LOCK.json")
    labels = demo_labels()
    assert b.fingerprint(labels) == run_lock["demo_label_prefix_sha256"]
    support = b.rows("support")
    probabilities = b.aggregate("support")
    truth = np.array([labels[int(row["target_center_s"] * 25)] for row in support])
    train = np.array([row["block"] == "threshold" for row in support])
    thresholds = alignment.fit_thresholds(probabilities[train], truth[train], "balanced")
    emissions = ensemble.threshold_emissions(thresholds, probabilities[~train])
    no_temporal = alignment.decode_threshold(probabilities[~train], thresholds)
    assert np.array_equal(no_temporal, temporal.viterbi(emissions, 0, 0))
    scores = {
        parameter: alignment.fast_metrics(truth[~train], temporal.viterbi(emissions, *parameter))
        for parameter in temporal.PARAMETERS
    }
    penalties = max(scores, key=lambda parameter: temporal.score_key(scores[parameter], parameter))
    return {
        "thresholds": thresholds.tolist(), "change_penalty": penalties[0], "order_penalty": penalties[1],
        "temporal_selection_score": scores[penalties], "temporal_no_prior_score": scores[(0.0, 0.0)],
        "labels_by_block": {
            name: dict(Counter(b.LABELS[label] for row, label in zip(support, truth) if row["block"] == name))
            for name in BLOCKS
        },
        "raw_support_sha256": {str(path): b.sha(path) for path in sorted(ROOT.glob("raw/support_rank*.jsonl"))},
    }


def select():
    path = ROOT / "LOCK.json"
    if path.exists():
        raise FileExistsError("The calibration is already frozen")
    b.check(b.read(ROOT / "RUN_LOCK.json"))
    assert not b.outputs("query"), "Query inference began before calibration"
    b.save(path, {"created_utc": now(), **fit(), "query_labels_used": False})


def predict():
    lock = b.read(ROOT / "LOCK.json")
    b.check({"source_sha256": lock["raw_support_sha256"]})
    rows = b.rows("query")
    mean = b.aggregate("query")
    raw = b.outputs("query")
    thresholds = np.asarray(lock["thresholds"])
    arms = {
        f"{bank}_raw": np.argmax(
            [raw[f'query:{row["video_id"]}:{row["window_id"]}:{bank}']["probabilities"] for row in rows], axis=1
        )
        for bank in BANKS
    }
    arms["dual_raw"] = np.argmax(mean, axis=1)
    arms["dual_thresholds"] = alignment.decode_threshold(mean, thresholds)
    videos = np.asarray([row["video_id"] for row in rows])
    arms["dual_soft_temporal"] = temporal.decode_videos(
        thresholds, mean, rows, videos, (lock["change_penalty"], lock["order_penalty"]),
        emission_function=ensemble.threshold_emissions,
    )
    return rows, arms


def historical():
    path = b.OLD / "temporal_prior_v1/evaluation.json"
    return b.read(path)["query_diagnostics_after_lock"]


def evaluate():
    b.check(b.read(ROOT / "RUN_LOCK.json"))
    rows, arms = predict()
    predictions = {name: pred.tolist() for name, pred in arms.items()}
    b.save(ROOT / "query_predictions.json", {
        "lock_sha256": b.sha(ROOT / "LOCK.json"),
        "frozen_utc": now(), "keys": [[row["video_id"], row["window_id"]] for row in rows],
        "predictions": predictions,
    })
    truth = b.truth(rows)
    b.save(ROOT / "evaluation.json", {
        "status": "complete", "predictions_sha256": b.sha(ROOT / "query_predictions.json"),
        "query_gt_used_only_for_scoring": True, "windows": len(rows),
        "query": {name: metrics(rows, pred, truth) for name, pred in arms.items()},
        "historical_many_video": historical(),
        "cost": {
            split: {"requests": len(outputs), "gpu_seconds": sum(out["elapsed_s"] for out in outputs.values())}
            for split in ("support", "query")
            for outputs in [b.outputs(split)]
        },
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "select", "evaluate"))
    args = parser.parse_args()
    globals()[args.stage]()
