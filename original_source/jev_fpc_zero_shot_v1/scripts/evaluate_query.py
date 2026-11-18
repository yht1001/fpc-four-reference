#!/usr/bin/env python3
"""Evaluate raw JEV decisions on the frozen six-video main-FPC query split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parents[1]
MANIFEST = OUT / "query" / "manifest.json"
LABELS = ("A01", "A02", "A03", "A04", "A05", "BG")
ACTION_LABELS = LABELS[:-1]
GT_TO_OPTION = {
    "取手机将其放到工作台": "A01",
    "将FPC贴在挤压板上": "A02",
    "使用挤压板粘贴FPC": "A03",
    "FPC卡扣扣合": "A04",
    "手机放回传送带上": "A05",
    "background": "BG",
}


def read_latest(path: Path) -> dict[tuple[str, int], dict]:
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") == "ok":
            latest[(str(row["video_id"]), int(row["window_id"]))] = row
    return latest


def classification_metrics(rows: list[dict]) -> dict:
    per_class = {}
    for label in LABELS:
        tp = sum(row["prediction"] == label and row["gt"] == label for row in rows)
        fp = sum(row["prediction"] == label and row["gt"] != label for row in rows)
        fn = sum(row["prediction"] != label and row["gt"] == label for row in rows)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_class[label] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
        }
    action_tp = sum(per_class[label]["tp"] for label in ACTION_LABELS)
    action_fp = sum(per_class[label]["fp"] for label in ACTION_LABELS)
    action_fn = sum(per_class[label]["fn"] for label in ACTION_LABELS)
    action_precision = action_tp / max(action_tp + action_fp, 1)
    action_recall = action_tp / max(action_tp + action_fn, 1)
    return {
        "samples": len(rows),
        "accuracy": sum(row["prediction"] == row["gt"] for row in rows) / max(len(rows), 1),
        "action_macro_f1": sum(per_class[label]["f1"] for label in ACTION_LABELS) / len(ACTION_LABELS),
        "action_micro_precision": action_precision,
        "action_micro_recall": action_recall,
        "action_micro_f1": 2 * action_precision * action_recall / max(action_precision + action_recall, 1e-12),
        "background_recall": per_class["BG"]["recall"],
        "prediction_counts": dict(Counter(row["prediction"] for row in rows)),
        "ground_truth_counts": dict(Counter(row["gt"] for row in rows)),
        "per_class": per_class,
    }


def get_segments(labels: list[str], remove_bg: bool) -> tuple[list[str], list[int], list[int]]:
    out_labels, starts, ends = [], [], []
    start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[start]:
            label = labels[start]
            if not (remove_bg and label == "BG"):
                out_labels.append(label)
                starts.append(start)
                ends.append(index)
            start = index
    return out_labels, starts, ends


def edit_score(pred: list[str], gt: list[str], remove_bg: bool) -> float:
    p, _, _ = get_segments(pred, remove_bg)
    g, _, _ = get_segments(gt, remove_bg)
    matrix = np.zeros((len(p) + 1, len(g) + 1), dtype=np.int64)
    matrix[:, 0] = np.arange(len(p) + 1)
    matrix[0, :] = np.arange(len(g) + 1)
    for i in range(1, len(p) + 1):
        for j in range(1, len(g) + 1):
            cost = 0 if p[i - 1] == g[j - 1] else 1
            matrix[i, j] = min(matrix[i - 1, j] + 1, matrix[i, j - 1] + 1, matrix[i - 1, j - 1] + cost)
    return 100.0 * (1.0 - matrix[-1, -1] / max(len(p), len(g), 1))


def segment_f1(pred: list[str], gt: list[str], overlap: float, remove_bg: bool) -> float:
    p_label, p_start, p_end = get_segments(pred, remove_bg)
    g_label, g_start, g_end = get_segments(gt, remove_bg)
    hits = np.zeros(len(g_label), dtype=bool)
    tp = fp = 0
    for label, start, end in zip(p_label, p_start, p_end):
        if not g_label:
            fp += 1
            continue
        intersection = np.minimum(end, np.asarray(g_end)) - np.maximum(start, np.asarray(g_start))
        union = np.maximum(end, np.asarray(g_end)) - np.minimum(start, np.asarray(g_start))
        iou = intersection / np.maximum(union, 1)
        iou *= np.asarray([label == value for value in g_label])
        best = int(np.argmax(iou))
        if iou[best] >= overlap and not hits[best]:
            tp += 1
            hits[best] = True
        else:
            fp += 1
    fn = len(g_label) - int(hits.sum())
    return 100.0 * 2 * tp / max(2 * tp + fp + fn, 1)


def temporal_metrics(rows: list[dict], remove_bg: bool) -> dict:
    pred = [row["prediction"] for row in rows]
    gt = [row["gt"] for row in rows]
    return {
        "Acc": 100.0 * sum(p == g for p, g in zip(pred, gt)) / max(len(gt), 1),
        "Edit": edit_score(pred, gt, remove_bg),
        "F1@10": segment_f1(pred, gt, 0.10, remove_bg),
        "F1@25": segment_f1(pred, gt, 0.25, remove_bg),
        "F1@50": segment_f1(pred, gt, 0.50, remove_bg),
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=OUT / "results" / "raw_predictions.jsonl")
    parser.add_argument("--results-dir", type=Path, default=OUT / "results")
    parser.add_argument("--prompt-layout", choices=("card_current", "visual_current"), default="card_current")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    predictions = read_latest(args.raw)
    all_rows = []
    by_video = {}
    missing = []
    for video in manifest["videos"]:
        gt_lines = [line.strip() for line in Path(video["gt_path"]).read_text(encoding="utf-8").splitlines()]
        rows = []
        for window in video["windows"]:
            key = (video["video_id"], int(window["window_id"]))
            if key not in predictions:
                missing.append(key)
                continue
            result = predictions[key]
            probabilities = {
                option: float(value)
                for option, value in result["response"]["answers"]["action"]["probabilities"].items()
            }
            choice = str(result["response"]["answers"]["action"]["choice"])
            center_frame = min(int(float(window["target_center_s"]) * video["fps"]), len(gt_lines) - 1)
            row = {
                "video_id": video["video_id"],
                "window_id": int(window["window_id"]),
                "target_start_s": window["target_start_s"],
                "target_end_s": window["target_end_s"],
                "target_center_s": window["target_center_s"],
                "prediction": choice,
                "gt": GT_TO_OPTION[gt_lines[center_frame]],
                "confidence": probabilities[choice],
                "probabilities": probabilities,
            }
            rows.append(row)
            all_rows.append(row)
        by_video[video["video_id"]] = rows
    if missing:
        raise SystemExit(f"incomplete inference: {len(missing)} missing, first={missing[:10]}")

    per_video = {}
    for video_id, rows in by_video.items():
        per_video[video_id] = {
            "classification": classification_metrics(rows),
            "temporal_with_background": temporal_metrics(rows, remove_bg=False),
            "temporal_without_background": temporal_metrics(rows, remove_bg=True),
        }
    temporal_macro = {}
    for policy, remove_bg in (("with_background", False), ("without_background", True)):
        metrics = [temporal_metrics(rows, remove_bg) for rows in by_video.values()]
        temporal_macro[policy] = {
            key: sum(item[key] for item in metrics) / len(metrics)
            for key in ("Acc", "Edit", "F1@10", "F1@25", "F1@50")
        }
        temporal_macro[policy]["FEA"] = sum(temporal_macro[policy].values()) / 5

    confusion = {
        gt: {pred: sum(row["gt"] == gt and row["prediction"] == pred for row in all_rows) for pred in LABELS}
        for gt in LABELS
    }
    evaluation = {
        "schema_version": "jev-fpc-zero-shot-evaluation-v1",
        "status": "complete" if len(all_rows) == manifest["total_windows"] else "incomplete",
        "protocol": {
            "split": "nfold split1: six query/test videos",
            "support_query_overlap": False,
            "model": "StartLux-Decision-v1.1-4B-AutoThink",
            "think_mode": "never",
            "prompt_layout": args.prompt_layout,
            "decision_rule": "raw argmax; no thresholds calibrated on query GT",
            "query_gt_used_for_inference_or_selection": False,
            "grid": "0.5 s stride; target-center label; coverage starts at 2.0 s",
        },
        "classification_aggregate": classification_metrics(all_rows),
        "temporal_macro_over_videos": temporal_macro,
        "confusion_matrix_gt_rows_pred_columns": confusion,
        "per_video": per_video,
    }
    results = args.results_dir
    results.mkdir(parents=True, exist_ok=True)
    (results / "predictions.json").write_text(json.dumps(all_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (results / "evaluation.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cls = evaluation["classification_aggregate"]
    temporal = evaluation["temporal_macro_over_videos"]
    class_lines = []
    for label in LABELS:
        item = cls["per_class"][label]
        class_lines.append(f"| {label} | {pct(item['precision'])} | {pct(item['recall'])} | {pct(item['f1'])} |")
    video_lines = []
    for video_id, value in per_video.items():
        item = value["classification"]
        video_lines.append(f"| `{video_id}` | {item['samples']} | {pct(item['accuracy'])} | {pct(item['action_macro_f1'])} | {pct(item['background_recall'])} |")
    report = f"""# 小米主FPC：跨视频 Query + Support JEV Zero-shot检测（{args.prompt_layout}）

## 正式协议

- Support：训练划分中的一条独立视频，A01–A05各三张early/middle/late参考图，另有三张BG参考图；
- Query：`nfold/split1.txt`冻结的6条测试视频，与support视频无重叠；
- 模型：`StartLux-Decision-v1.1-4B-AutoThink`，`think_mode=never`；
- Prompt：`{args.prompt_layout}`，模型看到不透明A01–A05/BG和参考图；`card_current`在option中加入动作摘要，`visual_current`只使用通用匹配套话；模型不看到中文GT类别名；
- 采样：每0.5秒判断一次，前2秒取2张context，最近0.5秒取3张target；
- 决策：六类概率直接argmax。Query GT没有参与prompt、阈值、support选择后的调参或实验臂选择；
- 评分：GT只在全部推理完成后读取；每个窗口以target中心帧标签评分，开头0–2秒不覆盖。

## 聚合分类结果

| 指标 | 结果 |
|---|---:|
| 窗口数 | {cls['samples']} |
| Accuracy | {pct(cls['accuracy'])} |
| 动作 Macro-F1 | {pct(cls['action_macro_f1'])} |
| 动作 Micro-F1 | {pct(cls['action_micro_f1'])} |
| BG Recall | {pct(cls['background_recall'])} |

预测分布：{json.dumps(cls['prediction_counts'], ensure_ascii=False)}。若预测集中在少数类别，
说明当前prompt与support存在输出偏置；这不等同于请求失败或GT标签映射错误。

## 各类别

| 类别 | Precision | Recall | F1 |
|---|---:|---:|---:|
{chr(10).join(class_lines)}

## 六条Query

| 视频 | 窗口数 | Accuracy | 动作Macro-F1 | BG Recall |
|---|---:|---:|---:|---:|
{chr(10).join(video_lines)}

## 标准时序指标（六视频宏平均）

| 口径 | Acc | Edit | F1@10 | F1@25 | F1@50 | FEA |
|---|---:|---:|---:|---:|---:|---:|
| 含BG | {temporal['with_background']['Acc']:.2f} | {temporal['with_background']['Edit']:.2f} | {temporal['with_background']['F1@10']:.2f} | {temporal['with_background']['F1@25']:.2f} | {temporal['with_background']['F1@50']:.2f} | {temporal['with_background']['FEA']:.2f} |
| 去BG | {temporal['without_background']['Acc']:.2f} | {temporal['without_background']['Edit']:.2f} | {temporal['without_background']['F1@10']:.2f} | {temporal['without_background']['F1@25']:.2f} | {temporal['without_background']['F1@50']:.2f} | {temporal['without_background']['FEA']:.2f} |
"""
    (results / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "status": evaluation["status"],
        "windows": cls["samples"],
        "accuracy": cls["accuracy"],
        "action_macro_f1": cls["action_macro_f1"],
        "background_recall": cls["background_recall"],
        "report": str((results / "REPORT.md").resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
