"""Extend the locked four-reference trial to the full FPC test split."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import hashlib
import importlib.util
import io
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / "four_worker_references_v1"
spec = importlib.util.spec_from_file_location("four_reference", PREVIOUS / "study.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.ROOT = ROOT
b = base.b
b.R = ROOT
all_inputs = b.inputs
decode = base.decode


def image_hashes(path):
    from PIL import Image

    data = path.read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        rgb = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
    return hashlib.sha256(data).hexdigest(), rgb


def frame_files(video):
    paths = sorted((b.FRAMES / video).glob("*.jpg"))
    assert paths and all(int(path.stem) == i for i, path in enumerate(paths, 1))
    return paths


def audit(manifest, previous):
    demo_ids = set(previous["references"].values())
    query_ids = {entry["video_id"] for entry in manifest}
    assert len(query_ids) == 6 and demo_ids.isdisjoint(query_ids)
    times = {video: tuple(datetime.fromisoformat(time) for time in pair)
             for video, pair in b.read(ROOT / "recording_times.json").items()}
    assert set(times) == demo_ids | query_ids
    frames = {video: frame_files(video) for video in sorted(times)}
    for video, (start, end) in times.items():
        assert abs((end - start).total_seconds() - (len(frames[video]) - 1) / 25) <= 1.1, video
    support = b.read(PREVIOUS / "inputs/support.json")["windows"]
    used_end = {video: max(row["target_end_s"] for row in support if row["source_video_id"] == video)
                for video in demo_ids}
    assert all(used_end[video] > base.VIDEO_LIMIT_S for video in demo_ids)

    source_hashes = ({}, {})
    duplicates = {}
    scan = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        for video in sorted(demo_ids) + sorted(query_ids):
            hits = {}
            digest = hashlib.sha256()
            for i, pair in enumerate(pool.map(image_hashes, frames[video])):
                for field, (table, value) in enumerate(zip(source_hashes, pair)):
                    digest.update(bytes.fromhex(value))
                    if video in demo_ids:
                        table.setdefault(value, (video, i))
                    elif value in table:
                        hits.setdefault(i, {})["jpeg" if field == 0 else "rgb"] = table[value]
            if video in query_ids:
                duplicates[video] = hits
            scan[video] = {"frames": len(frames[video]), "digest": digest.hexdigest()}
            print("Audited", video, len(frames[video]), "duplicate query frames", len(hits), flush=True)

    keys = ("video_id", "window_id", "target_start_s", "target_end_s", "target_center_s", "samples",
            "sheet", "sheet_sha256")
    rows, coverage = [], {}
    for item in manifest:
        video = item["video_id"]
        start, _ = times[video]
        related = [demo for demo in sorted(demo_ids) if base.recording_group(demo) == base.recording_group(video)]
        bad = set(duplicates[video])
        excluded = []
        for window in item["windows"]:
            left = min(sample["frame_index_1based"] - 1 for sample in window["samples"])
            right = math.ceil(window["target_end_s"] * 25 - 1e-7)
            assert 0 <= left < right <= len(frames[video])
            for sample in window["samples"]:
                assert Path(sample["source_path"]).resolve() == frames[video][sample["frame_index_1based"] - 1].resolve()
            content_overlap = not bad.isdisjoint(range(left, right))
            # The overlay is second-resolution. Allow a second on each bound when
            # comparing same-session clips against the pre-calibration label envelope.
            window_start = start + timedelta(seconds=left / 25 - 1)
            window_end = start + timedelta(seconds=right / 25 + 1)
            clock_overlap = any(
                window_start < times[source][0] + timedelta(seconds=used_end[source] + 1)
                and window_end > times[source][0] - timedelta(seconds=1)
                for source in related
            )
            if content_overlap or clock_overlap:
                excluded.append({"window_id": window["window_id"], "duplicate_frame": content_overlap,
                                 "overlaps_used_recording_time": clock_overlap})
                continue
            rows.append({key: window[key] for key in keys} | {"source_video_id": video})
        coverage[video] = {"total": len(item["windows"]), "excluded": excluded,
                           "same_recording_or_evening_batch": related}
    return rows, {"scan": scan, "windows_by_video": coverage, "duplicate_frames": duplicates,
                  "demo_label_and_input_envelope_end_s": used_end,
                  "v4_video": previous["chosen_query"][0]}


def prepare():
    assert not (ROOT / "RUN_LOCK.json").exists()
    parent = b.read(PREVIOUS / "RUN_LOCK.json")
    b.check(parent)
    assert b.read(PREVIOUS / "FINAL_VALIDATION.json")["status"] == "passed"
    manifest_path = base.FPC / "query/manifest.json"
    manifest = b.read(manifest_path)["videos"]
    assert sum(len(item["windows"]) for item in manifest) == 4786
    rows, audit_result = audit(manifest, b.read(PREVIOUS / "data_audit.json"))
    b.save(ROOT / "data_audit.json", audit_result)
    for name in ("LOCK.json", "REFERENCES.json"):
        shutil.copyfile(PREVIOUS / name, ROOT / name)
    for row in rows:
        row["request_sha256"] = {group: b.signature(row, group) for group in base.BANKS}
    b.save(ROOT / "inputs/query.json", {"windows": rows})
    v4 = audit_result["v4_video"]
    old_rows = {row["window_id"]: row for row in b.read(PREVIOUS / "inputs/query.json")["windows"]}
    cache_ids = {}
    for row in rows:
        if row["video_id"] == v4:
            assert old_rows[row["window_id"]] == row
            cache_ids.update({f'query:{v4}:{row["window_id"]}:{group}': digest
                              for group, digest in row["request_sha256"].items()})
    sources = sorted((PREVIOUS / "raw").glob("query_rank*.jsonl"))
    cache = [record for path in sources for record in b.jsonl(path) if record["id"] in cache_ids]
    assert {record["id"]: record["input_sha256"] for record in cache} == cache_ids
    assert len(cache) == len(cache_ids)
    archive = ROOT / "raw/query_rank_v4_cache.jsonl"
    archive.parent.mkdir(exist_ok=True)
    import json
    with archive.open("w") as out:
        for record in cache:
            out.write(json.dumps(record) + "\n")
    artifacts = [*ROOT.glob("*.py"), *ROOT.glob("*.json"), ROOT / "PROTOCOL.md", ROOT / "watermark_endpoints.png",
                 ROOT / "inputs/query.json", manifest_path, archive, PREVIOUS / "LOCK.json",
                 PREVIOUS / "FINAL_VALIDATION.json", PREVIOUS / "query_predictions.json", *sources]
    lock = {
        "source_sha256": parent["source_sha256"] | {str(path): b.sha(path) for path in artifacts},
        "weights_sha256": parent["weights_sha256"],
        "query_windows": len(rows), "original_windows": 4786,
        "cached_requests": len(cache), "new_requests": len(rows) * len(base.BANKS) - len(cache),
    }
    b.save(ROOT / "RUN_LOCK.json", lock)


def fresh_inputs(split):
    v4 = b.read(ROOT / "data_audit.json")["v4_video"]
    return [row for row in all_inputs(split) if row["video_id"] != v4]


def contiguous_rows(rows):
    result = []
    last_window = {}
    block = Counter()
    for row in rows:
        video, window = row["video_id"], row["window_id"]
        if video in last_window and window != last_window[video] + 1:
            block[video] += 1
        last_window[video] = window
        result.append(dict(row, video_id=f"{video}__{block[video]}" if block[video] else video))
    return result


def decode_contiguous(rows, probabilities, thresholds, penalties):
    return decode(contiguous_rows(rows), probabilities, thresholds, penalties)


def evaluate():
    from score import metrics

    b.check(b.read(ROOT / "RUN_LOCK.json"))
    rows, predictions = base.predict()
    keys = [[row["video_id"], row["window_id"]] for row in rows]
    v4 = b.read(ROOT / "data_audit.json")["v4_video"]
    indices = [i for i, (video, _) in enumerate(keys) if video == v4]
    original = b.read(PREVIOUS / "query_predictions.json")
    old_index = {tuple(key): i for i, key in enumerate(original["keys"])}
    assert not any(a != c for a, c in zip(rows, contiguous_rows(rows)) if a["video_id"] == v4)
    for arm, values in predictions.items():
        assert all(values[i] == original["predictions"][arm][old_index[tuple(keys[i])]] for i in indices)
    b.save(ROOT / "query_predictions.json", {
        "keys": keys, "lock_sha256": b.sha(ROOT / "LOCK.json"),
        "predictions": {arm: values.tolist() for arm, values in predictions.items()},
    })
    y = b.truth(rows)
    score_rows = contiguous_rows(rows)
    b.save(ROOT / "evaluation.json", {
        "status": "complete", "windows": len(rows), "videos": len({row["video_id"] for row in rows}),
        "prediction_sha256": b.sha(ROOT / "query_predictions.json"),
        "query": {arm: metrics(score_rows, values, y) for arm, values in predictions.items()},
    })


def validate():
    from validate_report import independent_temporal

    b.R = ROOT
    b.REFS = {bank: PREVIOUS / "references" / bank for bank in base.BANKS}
    lock = b.read(ROOT / "RUN_LOCK.json")
    b.check(lock)
    rows, predictions = base.predict()
    audit_result = b.read(ROOT / "data_audit.json")
    result = b.read(ROOT / "evaluation.json")
    saved = b.read(ROOT / "query_predictions.json")
    assert result["prediction_sha256"] == b.sha(ROOT / "query_predictions.json")
    assert result["windows"] == len(rows) == lock["query_windows"]
    assert saved["lock_sha256"] == b.sha(PREVIOUS / "LOCK.json")
    assert list(map(tuple, saved["keys"])) == [(row["video_id"], row["window_id"]) for row in rows]
    truth = b.truth(rows)
    by_span = b.np.array([row["video_id"] for row in contiguous_rows(rows)])
    for arm, values in predictions.items():
        assert saved["predictions"][arm] == values.tolist()
        c = result["query"][arm]["classification"]
        cm = b.np.zeros((6, 6), dtype=int)
        b.np.add.at(cm, (truth, values), 1)
        assert cm.tolist() == [[c["confusion"][gt][pred] for pred in b.LABELS] for gt in b.LABELS]
        hits = cm.diagonal()
        assert b.np.isclose(c["accuracy"], hits.sum() / len(truth))
        f1 = 2 * hits / b.np.maximum(1, cm.sum(0) + cm.sum(1))
        assert b.np.isclose(c["action_macro_f1"], f1[:5].mean())
        assert b.np.isclose(c["background_recall"], hits[5] / cm[5].sum())
        for span in set(by_span):
            keep = by_span == span
            for policy in ("with_background", "without_background"):
                actual = result["query"][arm]["per_video"][span]["temporal"][policy]
                independent = independent_temporal(truth[keep], values[keep], policy == "without_background")
                assert all(b.np.isclose(actual[k], score) for k, score in independent.items())
    excluded = sum(len(video["excluded"]) for video in audit_result["windows_by_video"].values())
    assert len(rows) + excluded == lock["original_windows"]
    write_report(result, audit_result, excluded)
    b.save(ROOT / "FINAL_VALIDATION.json", {
        "status": "passed", "query_videos": result["videos"], "query_windows": len(rows),
        "overlapping_windows_removed": excluded, "source_frames_scanned": sum(x["frames"] for x in audit_result["scan"].values()),
        "reused_v4_predictions_exact": True, "frozen_parameters_sha256": b.sha(ROOT / "LOCK.json"),
        "query_gt_used_only_after_prediction_freeze": True,
    })


def write_report(result, data, excluded):
    lines = [
        "# 小米 FPC 四参考六条 Query", "",
        f"原 4,786 个中心窗中排除 {excluded} 个和 Demo 源素材重合的窗口，保留 {result['videos']} 条视频、{result['windows']:,} 窗。",
        "V2/V3 同原始录像的不同片段按水印和全帧内容核查。V1/V6 同外观组或录制批次。此表用于跨片段比较。V4 是前一轮录像隔离子集。",
        "直接复用前一轮锁定阈值与离线软时序参数。V4 重用相同请求的原始概率，其他视频新增推理。", "",
        "| 参考＋离线软时序 | Accuracy | 动作 Macro-F1 | FG Recall | BG Recall | 去 BG F1@50 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, banks in base.ARMS.items():
        score = result["query"][f"{arm}_temporal"]
        c = score["classification"]
        lines.append(
            f"| {'＋'.join(banks)} | {c['accuracy']:.2%} | {c['action_macro_f1']:.2%} | "
            f"{c['foreground_recall']:.2%} | {c['background_recall']:.2%} | "
            f"{score['temporal']['without_background']['F1@50']:.2f} |"
        )
    lines += ["", "完整指标见 [evaluation.json](evaluation.json)，帧重合核查见 [data_audit.json](data_audit.json)。"]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("stage", choices=("prepare", "query", "evaluate", "validate"))
    cli.add_argument("rank", nargs="?", type=int)
    cli.add_argument("world", nargs="?", type=int)
    args = cli.parse_args()
    base.decode = decode_contiguous
    if args.stage == "query":
        b.inputs = fresh_inputs
        from infer import main

        main("query", args.rank, args.world)
    else:
        {"prepare": prepare, "evaluate": evaluate, "validate": validate}[args.stage]()
