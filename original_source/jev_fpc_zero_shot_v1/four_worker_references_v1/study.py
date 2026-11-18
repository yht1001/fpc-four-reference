"""Calibrate and evaluate four-reference and paired two-reference ensembles."""

import argparse
from collections import Counter
import csv
import hashlib
import io
from itertools import combinations, groupby
from pathlib import Path
import re
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
FPC = ROOT.parent
DATA = ROOT.parents[1] / "datasets/FPCdataset2601"
MATERIALS = ROOT.parents[1] / "组会汇报/FPC_工人素材核查"
ARCHIVE = MATERIALS / "FPC_六组动作参考_58.68秒.zip"
CHAPTERS = MATERIALS / "完整视频_章节索引.csv"
AUTO = FPC / "single_demo_dualref_v1"
BANKS = ("V1", "V2", "V3", "V6")
ARMS = {"four": BANKS, **{"_".join(pair): pair for pair in combinations(BANKS, 2)}}
BLOCKS = ("threshold", "temporal")
VIDEO_LIMIT_S = 120

sys.path.insert(0, str(FPC / "bin02_dualref_v1"))
import bin02_env as b

b.R = ROOT
b.REFS = {bank: ROOT / "references" / bank for bank in BANKS}
sys.path.append(str(AUTO))


def recording_group(video):
    if video.startswith("Pro"):
        return "2026-01-06-evening"
    return re.sub(r"(?:_part\d+(?:_480p)?|_\d{3}|_480p)$", "", video)


def get_labels(video):
    from evaluate_query import GT_TO_OPTION

    return [GT_TO_OPTION[line.strip()] for line in (b.GT / f"{video}.txt").read_text().splitlines()]


def source_inputs():
    with CHAPTERS.open(encoding="utf-8-sig") as stream:
        chapters = {row["视频ID"]: row["外观组"] for row in csv.DictReader(stream)}
    with ZipFile(ARCHIVE) as archive:
        with io.TextIOWrapper(archive.open("参考片段_来源.csv"), encoding="utf-8-sig") as stream:
            sources = {row["外观组"]: row["原视频ID"] for row in csv.DictReader(stream)
                       if row["外观组"] in BANKS}
        with io.TextIOWrapper(archive.open("参考片段_动作标注.csv"), encoding="utf-8-sig") as stream:
            annotations = [row for row in csv.DictReader(stream) if row["外观组"] in BANKS]
    assert set(sources) == set(BANKS) and len(set(sources.values())) == 4
    for bank, video in sources.items():
        assert chapters[video] == bank
    splits = DATA / "split/nfold"
    main_ids = set((splits / "split1.txt").read_text().split())
    support_ids = {video for fold in (2, 3, 4) for video in (splits / f"split{fold}.txt").read_text().split()}
    assert set(sources.values()) <= support_ids - main_ids
    query_ids = {video for video in main_ids if chapters[video] == "V4"}
    assert len(query_ids) == 1
    assert {recording_group(video) for video in query_ids}.isdisjoint(map(recording_group, sources.values()))
    return sources, chapters, annotations, query_ids, main_ids - query_ids


def background_spans(labels, actions):
    inside = [(left[1], right[0]) for left, right in zip(actions, actions[1:]) if left[1] < right[0]]
    assert all(set(labels[start:stop]) == {"BG"} for start, stop in inside)
    inside = sorted(sorted(inside, key=lambda span: (span[0] - span[1], span[0]))[:2])
    assert len(inside) == 2

    foreground = []
    start = 0
    for label, run in groupby(labels[:VIDEO_LIMIT_S * 25]):
        stop = start + sum(1 for _ in run)
        if label != "BG" and stop < VIDEO_LIMIT_S * 25:
            foreground.append((label, start, stop))
        start = stop
    expected = b.LABELS[:-1] * 2
    gaps = []
    for index in range(len(foreground) - 9):
        events = foreground[index:index + 10]
        if [event[0] for event in events] == expected:
            gap = events[4][2], events[5][1]
            if gap[0] < gap[1]:
                assert set(labels[gap[0]:gap[1]]) == {"BG"}
                gaps.append(gap)
    center = sum((actions[0][0], actions[-1][1]))
    inter = min(gaps, key=lambda gap: (
        gap[0] != actions[-1][1] and gap[1] != actions[0][0], abs(sum(gap) - center), gap[0]
    ))
    return inside + [inter]


def prepare_refs(sources, annotations):
    from build_support import make_sheet

    references = {}
    for bank, video in sources.items():
        labels = get_labels(video)
        action_spans = {}
        for row in annotations:
            if row["外观组"] == bank and row["类别"] != "BG":
                assert row["原视频ID"] == video and row["类别"] not in action_spans
                action_spans[row["类别"]] = tuple(
                    round(float(row[key]) * 25) for key in ("原视频起点(s)", "原视频终点(s)")
                )
        actions = [action_spans[label] for label in b.LABELS[:-1]]
        assert all(left[1] <= right[0] for left, right in zip(actions, actions[1:]))
        backgrounds = background_spans(labels, actions)
        references[bank] = {}
        for label, span in action_spans.items():
            start, stop = span
            frames = [start + (stop - start) * fraction // 10 for fraction in (2, 5, 8)]
            assert len(set(frames)) == 3 and set(labels[start:stop]) == {label}
            references[bank][label] = {
                "frames": frames, "spans": [span], "kinds": ["early", "middle", "late"],
            }
        references[bank]["BG"] = {
            "frames": [(start + stop) // 2 for start, stop in backgrounds],
            "spans": backgrounds, "kinds": ["between_actions", "between_actions", "between_cycles"],
        }
        for label, ref in references[bank].items():
            assert all(0 <= frame < VIDEO_LIMIT_S * 25 and labels[frame] == label for frame in ref["frames"])
            paths = [b.FRAMES / video / f"{frame + 1:06d}.jpg" for frame in ref["frames"]]
            path = b.REFS[bank] / f"{label}.jpg"
            make_sheet(paths, path)
            ref.update(video_id=video, sheet=str(path), sheet_sha256=b.sha(path), paths=list(map(str, paths)))
    return references


def audit_frames(references, support, query):
    from PIL import Image

    def sample_path(row, sample):
        path = b.FRAMES / row["source_video_id"] / f'{sample["frame_index_1based"]:06d}.jpg'
        assert path.resolve() == Path(sample["source_path"]).resolve()
        return str(path)

    groups = {"reference": {p for bank in references.values() for ref in bank.values() for p in ref["paths"]}}
    for split in (*BLOCKS, "query"):
        windows = query if split == "query" else [row for row in support if row["block"] == split]
        groups[split] = {sample_path(row, sample) for row in windows for sample in row["samples"]}
        for row in windows:
            assert b.sha(row["sheet"]) == row["sheet_sha256"]
            if split in BLOCKS:
                start, stop = row["calibration_bounds_s"]
                assert start - 2 >= VIDEO_LIMIT_S and stop - start == 20
                assert start <= row["target_start_s"] < row["target_end_s"] <= stop
                assert all((start - 2) * 25 <= s["frame_index_1based"] - 1 < stop * 25 for s in row["samples"])
    digests = {}
    for group, paths in groups.items():
        digests[group] = {}
        for path in sorted(paths):
            with Image.open(path) as frame:
                pixel_hash = hashlib.sha256(frame.convert("RGB").tobytes()).hexdigest()
            digests[group][path] = {"jpeg": b.sha(path), "rgb": pixel_hash}
    for left, right in combinations(groups, 2):
        assert groups[left].isdisjoint(groups[right]), (left, right)
        for field in ("jpeg", "rgb"):
            left_hash = {record[field] for record in digests[left].values()}
            right_hash = {record[field] for record in digests[right].values()}
            assert left_hash.isdisjoint(right_hash), (left, right, field)
    return digests


def calibration_block(labels, earliest_start):
    for start in range(earliest_start, len(labels) // 25 - 19, 5):
        centers = [int((start + .25 + .5 * i) * 25) for i in range(40)]
        if {labels[index] for index in centers} == set(b.LABELS):
            return start, start + 20
    raise ValueError("No complete labeled 20 s calibration block")


def prepare():
    import build_threshold_calibration as calibration

    if (ROOT / "RUN_LOCK.json").exists():
        raise FileExistsError("The input lock already exists")
    sources, chapters, annotations, query_ids, excluded = source_inputs()
    refs = prepare_refs(sources, annotations)
    b.save(ROOT / "reference_provenance.json", refs)
    b.save(ROOT / "REFERENCES.json", {
        bank: [options[label]["sheet_sha256"] for label in b.LABELS] for bank, options in refs.items()
    })
    calibration.DESTINATION = ROOT
    support = []
    gt_hashes = {}
    for bank, video in sources.items():
        labels = get_labels(video)
        gt_hashes[str(b.GT / f"{video}.txt")] = b.sha(b.GT / f"{video}.txt")
        threshold = calibration_block(labels, 125)
        temporal = calibration_block(labels, max(155, threshold[1] + 5))
        for block, bounds in zip(BLOCKS, (threshold, temporal)):
            start, stop = bounds
            role = f"{bank}_{block}"
            rows = calibration.build_split(video, role, start + .25, stop - .25, len(labels))["windows"]
            for row in rows:
                row.update(block=block, calibration_bounds_s=bounds)
            assert {labels[int(row["target_center_s"] * 25)] for row in rows} == set(b.LABELS), role
            support.extend(rows)
    query_manifest = FPC / "query/manifest.json"
    query_keys = ("video_id", "window_id", "target_start_s", "target_end_s", "target_center_s", "samples", "sheet", "sheet_sha256")
    query = [
        {key: window[key] for key in query_keys} | {"source_video_id": item["video_id"]}
        for item in b.read(query_manifest)["videos"] if item["video_id"] in query_ids
        for window in item["windows"]
    ]
    assert len(query) == 799 and len(support) == 320
    frame_hashes = audit_frames(refs, support, query)
    b.save(ROOT / "data_audit.json", {
        "references": sources, "chosen_query": sorted(query_ids),
        "excluded_query": {video: {"appearance_group": chapters[video], "recording_group": recording_group(video)}
                           for video in sorted(excluded)},
        "exact_frame_path_jpeg_and_rgb_intersections": 0, "frame_hashes": frame_hashes,
        "identity_status": "User confirmed V1/V2/V3/V6 distinct; V4 independent recording, identity unconfirmed",
    })
    for split, rows in (("support", support), ("query", query)):
        for row in rows:
            row["request_sha256"] = {group: b.signature(row, group) for group in BANKS}
        b.save(ROOT / "inputs" / f"{split}.json", {"windows": rows})

    source_files = [
        *ROOT.glob("*.py"), ROOT / "PROTOCOL.md", ROOT / "data_audit.json", ROOT / "REFERENCES.json",
        ROOT / "reference_provenance.json", *ROOT.glob("inputs/*.json"), *ROOT.glob("references/*/*.jpg"),
        ARCHIVE, CHAPTERS, query_manifest, *(DATA / "split/nfold").glob("split*.txt"),
        AUTO / "validate_report.py",
    ]
    previous = b.read(AUTO / "RUN_LOCK.json")
    code = {path: digest for path, digest in previous["source_sha256"].items() if Path(path).suffix == ".py"}
    b.save(ROOT / "RUN_LOCK.json", {
        "source_sha256": code | gt_hashes | {str(path): b.sha(path) for path in source_files},
        "weights_sha256": previous["weights_sha256"], "support_windows": len(support),
        "query_windows": len(query), "banks": BANKS, "arms": ARMS,
    })


def probability_cube(split):
    import numpy as np

    b.aggregate(split)
    outputs = b.outputs(split)
    return np.asarray([
        [outputs[f'{split}:{row["video_id"]}:{row["window_id"]}:{bank}']["probabilities"] for bank in BANKS]
        for row in b.rows(split)
    ])


def mixture(cube, banks):
    return cube[:, [BANKS.index(bank) for bank in banks]].mean(axis=1)


def decode(rows, probabilities, thresholds, penalties):
    import numpy as np
    from evaluate_ensemble_temporal_prior import threshold_emissions
    from evaluate_soft_temporal_prior import decode_videos

    return decode_videos(np.asarray(thresholds), probabilities, rows,
                         np.array([row["video_id"] for row in rows]), tuple(penalties),
                         emission_function=threshold_emissions)


def fit():
    import numpy as np
    from evaluate_probability_alignment import decode_threshold, fast_metrics, fit_thresholds
    from evaluate_soft_temporal_prior import PARAMETERS, score_key

    rows = b.rows("support")
    assert {row["source_video_id"] for row in rows} == set(b.read(ROOT / "data_audit.json")["references"].values())
    labels = {video: get_labels(video) for video in {row["source_video_id"] for row in rows}}
    y = np.array([b.LABELS.index(labels[row["source_video_id"]][int(row["target_center_s"] * 25)]) for row in rows])
    train = np.array([row["block"] == "threshold" for row in rows])
    holdout = [row for row, use in zip(rows, train) if not use]
    cube = probability_cube("support")
    fitted = {}
    for arm, banks in ARMS.items():
        p = mixture(cube, banks)
        thresholds = fit_thresholds(p[train], y[train], "balanced")
        assert np.array_equal(decode_threshold(p[~train], thresholds), decode(holdout, p[~train], thresholds, (0, 0)))
        scores = {pair: fast_metrics(y[~train], decode(holdout, p[~train], thresholds, pair)) for pair in PARAMETERS}
        best = max(scores, key=lambda pair: score_key(scores[pair], pair))
        fitted[arm] = {"banks": list(banks), "thresholds": thresholds.tolist(), "penalties": list(best),
                       "temporal_selection_score": scores[best]}
    return fitted


def select():
    assert not (ROOT / "LOCK.json").exists() and not b.outputs("query")
    b.check(b.read(ROOT / "RUN_LOCK.json"))
    b.save(ROOT / "LOCK.json", {
        "created_utc": b.datetime.datetime.now(b.datetime.timezone.utc).isoformat(),
        "arms": fit(), "query_gt_used": False,
        "support_raw_sha256": {str(path): b.sha(path) for path in ROOT.glob("raw/support_rank*.jsonl")},
    })


def predict():
    from evaluate_probability_alignment import decode_threshold

    lock = b.read(ROOT / "LOCK.json")
    b.check({"source_sha256": lock["support_raw_sha256"]})
    rows, cube = b.rows("query"), probability_cube("query")
    pred = {}
    for arm, fitted in lock["arms"].items():
        p = mixture(cube, fitted["banks"])
        pred[f"{arm}_raw"] = p.argmax(axis=1)
        pred[f"{arm}_threshold"] = decode_threshold(p, b.np.asarray(fitted["thresholds"]))
        pred[f"{arm}_temporal"] = decode(rows, p, fitted["thresholds"], fitted["penalties"])
    return rows, pred


def evaluate():
    from score import metrics

    b.check(b.read(ROOT / "RUN_LOCK.json"))
    rows, pred = predict()
    b.save(ROOT / "query_predictions.json", {
        "keys": [[row["video_id"], row["window_id"]] for row in rows],
        "lock_sha256": b.sha(ROOT / "LOCK.json"),
        "predictions": {arm: values.tolist() for arm, values in pred.items()},
    })
    y = b.truth(rows)
    b.save(ROOT / "evaluation.json", {
        "status": "complete", "windows": len(rows), "prediction_sha256": b.sha(ROOT / "query_predictions.json"),
        "query": {arm: metrics(rows, values, y) for arm, values in pred.items()},
    })


def validate():
    import numpy as np
    # Importing the archived validator binds its own experiment paths.
    backend_path, references = b.R, b.REFS
    from validate_report import independent_temporal

    b.R, b.REFS = backend_path, references

    b.check(b.read(ROOT / "RUN_LOCK.json"))
    lock = b.read(ROOT / "LOCK.json")
    assert not lock["query_gt_used"] and lock["arms"] == fit()
    rows, pred = predict()
    saved = b.read(ROOT / "query_predictions.json")
    result = b.read(ROOT / "evaluation.json")
    assert saved["lock_sha256"] == b.sha(ROOT / "LOCK.json")
    assert result["prediction_sha256"] == b.sha(ROOT / "query_predictions.json")
    assert saved["keys"] == [[row["video_id"], row["window_id"]] for row in rows]
    assert audit_frames(b.read(ROOT / "reference_provenance.json"), b.rows("support"), rows) == (
        b.read(ROOT / "data_audit.json")["frame_hashes"]
    )
    truth = b.truth(rows)
    assert len({row["video_id"] for row in rows}) == 1
    for arm, values in pred.items():
        assert saved["predictions"][arm] == values.tolist()
        c = result["query"][arm]["classification"]
        counts = Counter(zip(truth, values))
        assert all(c["confusion"][gt][p] == counts[i, j] for i, gt in enumerate(b.LABELS) for j, p in enumerate(b.LABELS))
        confusion = np.array([[counts[i, j] for j in range(6)] for i in range(6)])
        hits = confusion.diagonal()
        f1 = 2 * hits / np.maximum(1, confusion.sum(0) + confusion.sum(1))
        assert np.isclose(c["accuracy"], hits.sum() / len(truth))
        assert np.isclose(c["action_macro_f1"], f1[:5].mean())
        assert np.isclose(c["background_recall"], hits[5] / confusion[5].sum())
        assert np.isclose(c["foreground_recall"], confusion[:5, :5].sum() / confusion[:5].sum())
        for policy in ("with_background", "without_background"):
            actual = result["query"][arm]["temporal"][policy]
            expected = independent_temporal(truth, values, policy == "without_background")
            assert all(np.isclose(actual[k], value) for k, value in expected.items())
    write_report(result, lock)
    b.save(ROOT / "FINAL_VALIDATION.json", {
        "status": "passed", "reference_frames": 72, "support_windows": 320, "query_windows": len(rows),
        "calibration_exact_replay": "passed", "source_frame_paths_jpeg_and_rgb_overlap": 0,
        "independent_confusion_and_temporal_metrics": "passed", "query_gt_used_only_after_prediction_freeze": True,
    })


def write_report(result, lock):
    lines = [
        "# 小米 FPC 四工人参考与双参考对照", "",
        "V1、V2、V3、V6 各提供一套参考。每动作三个实例内面板，每背景三面板覆盖两个动作间过渡与一个循环间间隙。",
        "校准使用同四条录像中与参考分离的两段时窗。按录像来源和批次剔除五条旧 Query，保留 V4 的 799 窗。V4 工人身份仍未确认。",
        "冻结 StartLux v1.1。所有组合等权概率融合，分别按同一规则重新校准。V1＋V2 是事前指定的双参考对照。", "",
        "| 参考＋软时序 | Accuracy | 动作 Macro-F1 | FG Recall | BG Recall | 去 BG F1@50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, banks in ARMS.items():
        scores = result["query"][f"{arm}_temporal"]
        c = scores["classification"]
        name = "＋".join(banks)
        lines.append(
            f"| {name} | {c['accuracy']:.2%} | {c['action_macro_f1']:.2%} | "
            f"{c['foreground_recall']:.2%} | {c['background_recall']:.2%} | "
            f"{scores['temporal']['without_background']['F1@50']:.2f} |"
        )
    lines += [
        "", "另有不加时序的分类、完整混淆矩阵、两种背景口径的时序指标，见 [evaluation.json](evaluation.json)。",
        "各组校准阈值及惩罚在 [LOCK.json](LOCK.json)。帧与拆分审计在 [data_audit.json](data_audit.json)。",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "support", "query", "select", "evaluate", "validate"))
    parser.add_argument("rank", type=int, nargs="?")
    parser.add_argument("world", type=int, nargs="?")
    args = parser.parse_args()
    if args.stage in {"support", "query"}:
        from infer import main

        if args.stage == "query":
            b.check({"source_sha256": b.read(ROOT / "LOCK.json")["support_raw_sha256"]})
        main(args.stage, args.rank, args.world)
    else:
        {"prepare": prepare, "select": select, "evaluate": evaluate, "validate": validate}[args.stage]()
