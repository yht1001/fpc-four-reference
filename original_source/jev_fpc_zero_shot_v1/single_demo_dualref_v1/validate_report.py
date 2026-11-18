"""Audit single-Demo provenance and independently check the reported outcomes."""

import datetime
from itertools import groupby

import numpy as np

from experiment import BANKS, BLOCKS, DEMO, ROOT, b, demo_labels, fit, predict, recording_id


def segment_list(labels, remove_bg):
    return [
        (int(label), start, stop)
        for label, start, stop in binned_segments(labels)
        if label != 5 or not remove_bg
    ]


def binned_segments(labels):
    start = 0
    for label, group in groupby(labels):
        stop = start + sum(1 for _ in group)
        yield label, start, stop
        start = stop


def independent_temporal(y, prediction, remove_bg):
    reference = segment_list(y, remove_bg)
    decoded = segment_list(prediction, remove_bg)
    distance = list(range(len(reference) + 1))
    for i, (label, _, _) in enumerate(decoded, 1):
        previous = distance
        distance = [i]
        for j, (true_label, _, _) in enumerate(reference, 1):
            distance.append(min(previous[j] + 1, distance[-1] + 1, previous[j - 1] + (label != true_label)))
    scores = {"Edit": 100 * (1 - distance[-1] / max(len(reference), len(decoded), 1))}
    for threshold in (10, 25, 50):
        matched = set()
        for label, start, stop in decoded:
            overlaps = [
                max(0, min(stop, end) - max(start, begin)) / (max(stop, end) - min(start, begin))
                if label == true_label else 0
                for true_label, begin, end in reference
            ]
            if overlaps:
                best = int(np.argmax(overlaps))
                if best not in matched and 100 * overlaps[best] >= threshold:
                    matched.add(best)
        scores[f"F1@{threshold}"] = 200 * len(matched) / max(len(reference) + len(decoded), 1)
    return scores


def main():
    run = b.read(ROOT / "RUN_LOCK.json")
    lock = b.read(ROOT / "LOCK.json")
    b.check(run)
    assert all(lock[key] == value for key, value in fit().items())
    demo = b.read(ROOT / "DEMO_LOCK.json")
    assert demo["video_id"] == DEMO and demo["label_frame_range_zero_based"] == [0, 1500]
    assert not lock["query_labels_used"]
    labels = demo_labels()
    refs = b.read(ROOT / "reference_provenance.json")
    assert list(refs) == list(BANKS)
    assert refs[BANKS[0]]["cycle"]["end_s"] <= refs[BANKS[1]]["cycle"]["start_s"]
    for bank in refs.values():
        for label, item in bank["labels"].items():
            b.check({"source_sha256": item["source_sha256"]})
            assert {b.LABELS[labels[index]] for index in item["source_frames_zero_based"]} == {label}
            assert all(index < 750 for index in item["source_frames_zero_based"])
            assert all(b.FRAMES.resolve() / DEMO == b.Path(path).resolve().parent for path in item["source_sha256"])
            assert b.sha(item["sheet"]) == item["sheet_sha256"]
    support = b.rows("support")
    assert {row["video_id"] for row in support} == {DEMO}
    for row in support:
        start, stop = BLOCKS[row["block"]]
        assert start <= row["target_start_s"] < row["target_end_s"] <= stop
        assert all(b.Path(sample["source_path"]).parent.name == DEMO for sample in row["samples"])
        assert b.sha(row["sheet"]) == row["sheet_sha256"]

    rows, arms = predict()
    predictions = b.read(ROOT / "query_predictions.json")
    report = b.read(ROOT / "evaluation.json")
    assert predictions["lock_sha256"] == b.sha(ROOT / "LOCK.json")
    assert report["predictions_sha256"] == b.sha(ROOT / "query_predictions.json")
    assert datetime.datetime.fromisoformat(demo["created_utc"]) <= datetime.datetime.fromisoformat(lock["created_utc"]) <= datetime.datetime.fromisoformat(predictions["frozen_utc"])
    assert predictions["keys"] == [[row["video_id"], row["window_id"]] for row in rows]
    assert len(rows) == 4786 and len(demo["query_ids"]) == 6
    assert {row["video_id"] for row in rows} == set(demo["query_ids"])
    assert recording_id(DEMO) not in {recording_id(video) for video in demo["query_ids"]}
    truth = b.truth(rows)
    videos = np.array([row["video_id"] for row in rows])
    for name, values in arms.items():
        assert predictions["predictions"][name] == values.tolist()
        counts = np.zeros((6, 6), dtype=int)
        np.add.at(counts, (truth, values), 1)
        tps = counts.diagonal()
        classification = report["query"][name]["classification"]
        assert counts.tolist() == [[classification["confusion"][gt][pred] for pred in b.LABELS] for gt in b.LABELS]
        assert np.isclose(classification["accuracy"], np.mean(values == truth))
        assert np.isclose(classification["action_macro_f1"], (2 * tps / np.maximum(1, counts.sum(0) + counts.sum(1)))[:5].mean())
        assert np.isclose(classification["background_recall"], counts[5, 5] / counts[5, :].sum())
        for policy, drop in (("with_background", False), ("without_background", True)):
            for video in demo["query_ids"]:
                keep = videos == video
                scores = independent_temporal(truth[keep], values[keep], drop)
                recorded = report["query"][name]["per_video"][video]["temporal"][policy]
                assert all(np.isclose(recorded[metric], value) for metric, value in scores.items()), (name, policy, video, scores, recorded)

    b.save(ROOT / "FINAL_VALIDATION.json", {
        "status": "passed", "verified_reference_frames": 36, "verified_support_windows": len(support),
        "verified_query_windows": len(rows), "source_and_prediction_hashes": "passed",
        "support_only_calibration_exact_replay": "passed", "independent_confusion_edit_and_segment_f1": "passed",
    })
    write_report(report, lock, refs)


def write_report(result, lock, refs):
    lines = [
        "# 小米 FPC 单条 Demo＋双参考＋软时序",
        "",
        f"固定 Demo 为 `{DEMO}`。参考、阈值及软时序只用该视频前 60 秒的已有帧级标注。",
        "两套参考各使用 1 个五动作循环，共 10 个动作实例和 6 张背景取样帧。",
        "阈值和时序各用 26 个不相交的中心窗，分别来自同一 Demo 的 32–45 秒和 47–60 秒。",
        "冻结原 StartLux v1.1 权重，沿用历史 0.5 秒输入和评分口径。6 条历史 Query 共 4,786 窗，与 Demo 的文件名前缀均不重合，Query GT 仅评分。",
        "该设置是按视频计数的单 Demo 迁移。历史 Query 已在以往实验中评测过。",
        "",
        "| 方案 | Accuracy | 动作 Macro-F1 | BG Recall | FG Recall | 去 BG Edit | 去 BG F1@10/25/50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = {
        "cycle_a_raw": "单 Demo 参考 A raw", "cycle_b_raw": "单 Demo 参考 B raw",
        "dual_raw": "单 Demo 双参考 raw", "dual_thresholds": "单 Demo 双参考＋阈值",
        "dual_soft_temporal": "单 Demo 双参考＋阈值＋软时序",
    }
    for name, score in result["query"].items():
        c, t = score["classification"], score["temporal"]["without_background"]
        lines.append(
            f"| {names[name]} | {c['accuracy']:.2%} | {c['action_macro_f1']:.2%} | "
            f"{c['background_recall']:.2%} | {c['foreground_recall']:.2%} | {t['Edit']:.2f} | "
            f"{t['F1@10']:.2f}/{t['F1@25']:.2f}/{t['F1@50']:.2f} |"
        )
    old = result["historical_many_video"]["soft_temporal_prior"]
    lines += [
        "",
        f"多视频归档基线的 Accuracy、动作 Macro-F1 和 BG Recall 为 {old['accuracy']:.2%}、{old['action_macro_f1']:.2%} 和 {old['background_recall']:.2%}。",
        "归档基线与本轮的参考和校准标注量同时不同。",
        "",
        f"校准阈值 A01–A05/BG 为 {lock['thresholds']}，软时序切换和顺序惩罚为 {lock['change_penalty']} 和 {lock['order_penalty']}。",
        f"参考 A 使用 {refs[BANKS[0]]['cycle']['start_s']:.2f}–{refs[BANKS[0]]['cycle']['end_s']:.2f} 秒的循环，"
        f"参考 B 使用 {refs[BANKS[1]]['cycle']['start_s']:.2f}–{refs[BANKS[1]]['cycle']['end_s']:.2f} 秒的循环。",
        "",
        "完整分视频指标和混淆见 [evaluation.json](evaluation.json)，标注来源见 [DEMO_LOCK.json](DEMO_LOCK.json)，"
        "参数见 [LOCK.json](LOCK.json)，验证见 [FINAL_VALIDATION.json](FINAL_VALIDATION.json)。",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
