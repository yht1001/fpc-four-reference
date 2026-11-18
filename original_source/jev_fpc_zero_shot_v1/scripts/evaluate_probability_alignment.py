#!/usr/bin/env python3
"""Cross-validate FPC probability alignment and apply the locked method to Query."""

from __future__ import annotations

import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from evaluate_query import GT_TO_OPTION, LABELS, classification_metrics, temporal_metrics


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "probability_alignment_v1"
MANIFEST = EXP / "manifest.json"
RAW = EXP / "raw_predictions.jsonl"
FRAME_GT = ROOT.parent / "datasets" / "FPCdataset2601" / "annotation" / "gtTXT"
QUERY = ROOT / "results_visual_current" / "predictions.json"
QUERY_MANIFEST = ROOT / "query" / "manifest.json"
ACTION_LABELS = LABELS[:-1]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL = dict(enumerate(LABELS))
LEGACY_THRESHOLDS = {"A01": 0.10, "A02": 0.40, "A03": 0.20, "A04": 0.50, "A05": 0.50, "BG": 0.20}


def latest_results(path: Path) -> dict[tuple[str, int], dict]:
    latest = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            if row.get("status") == "ok":
                latest[(str(row["video_id"]), int(row["window_id"]))] = row
    return latest


def load_development_rows() -> list[dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = latest_results(RAW)
    rows = []
    missing = []
    for video in manifest["videos"]:
        gt = (FRAME_GT / f"{video['video_id']}.txt").read_text(encoding="utf-8").splitlines()
        for window in video["windows"]:
            key = (video["video_id"], int(window["window_id"]))
            if key not in results:
                missing.append(key)
                continue
            values = results[key]["response"]["answers"]["action"]["probabilities"]
            probabilities = {label: float(values[label]) for label in LABELS}
            center_frame = min(int(float(window["target_center_s"]) * float(video["fps"])), len(gt) - 1)
            rows.append({
                "video_id": video["video_id"],
                "session_group": video["session_group"],
                "window_id": int(window["window_id"]),
                "target_center_s": float(window["target_center_s"]),
                "gt": GT_TO_OPTION[gt[center_frame].strip()],
                "probabilities": probabilities,
            })
    if missing:
        raise SystemExit(f"incomplete development inference: {len(missing)} missing, first={missing[:10]}")
    return rows


def arrays(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probabilities = np.asarray([[row["probabilities"][label] for label in LABELS] for row in rows], dtype=np.float64)
    truth = np.asarray([LABEL_TO_INDEX[row["gt"]] for row in rows], dtype=np.int64)
    videos = np.asarray([row["video_id"] for row in rows], dtype=object)
    return probabilities, truth, videos


def fast_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    np.add.at(cm, (truth, prediction), 1)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / np.maximum(tp + fn, 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "action_macro_f1": float(f1[:5].mean()),
        "six_class_macro_f1": float(f1.mean()),
        "background_recall": float(recall[5]),
    }


def full_metrics(rows: list[dict], prediction: np.ndarray) -> dict:
    decoded = [{**row, "prediction": INDEX_TO_LABEL[int(value)]} for row, value in zip(rows, prediction)]
    metrics = classification_metrics(decoded)
    metrics["six_class_macro_f1"] = sum(metrics["per_class"][label]["f1"] for label in LABELS) / len(LABELS)
    return metrics


def threshold_array(values: dict[str, float]) -> np.ndarray:
    return np.asarray([values[label] for label in LABELS], dtype=np.float64)


def decode_threshold(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    winner = np.argmax(probabilities / thresholds[None, :], axis=1)
    below = (winner != 5) & (probabilities[np.arange(len(winner)), winner] < thresholds[winner])
    winner[below] = 5
    return winner


def fit_thresholds(probabilities: np.ndarray, truth: np.ndarray, objective: str) -> np.ndarray:
    grid = [round(0.10 + 0.05 * index, 2) for index in range(9)]
    best = None
    for values in itertools.product(grid, repeat=5):
        thresholds = np.asarray((*values, 0.20), dtype=np.float64)
        prediction = decode_threshold(probabilities, thresholds)
        score = fast_metrics(truth, prediction)
        regularizer = -sum(abs(value - 0.20) for value in values)
        if objective == "action_first":
            key = (score["action_macro_f1"], score["accuracy"], score["background_recall"], regularizer)
        elif objective == "balanced":
            key = (score["six_class_macro_f1"], score["action_macro_f1"], score["accuracy"],
                   score["background_recall"], regularizer)
        else:
            raise ValueError(objective)
        if best is None or key > best[0]:
            best = (key, thresholds.copy())
    assert best is not None
    return best[1]


def decode_foreground(probabilities: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.argmax(probabilities[:, :5] / thresholds[None, :], axis=1)


def fit_foreground_thresholds(probabilities: np.ndarray, truth: np.ndarray) -> np.ndarray:
    keep = truth != 5
    p = probabilities[keep]
    y = truth[keep]
    grid = [round(0.10 + 0.05 * index, 2) for index in range(9)]
    best = None
    for values in itertools.product(grid, repeat=5):
        thresholds = np.asarray(values, dtype=np.float64)
        prediction = decode_foreground(p, thresholds)
        score = fast_metrics(y, prediction)
        regularizer = -sum(abs(value - 0.20) for value in values)
        key = (score["action_macro_f1"], score["accuracy"], regularizer)
        if best is None or key > best[0]:
            best = (key, thresholds.copy())
    assert best is not None
    return best[1]


def gate_features(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-8, 1.0)
    actions = clipped[:, :5]
    sorted_actions = np.sort(actions, axis=1)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return np.column_stack([
        np.log(clipped[:, 5]),
        np.log(np.max(actions, axis=1)),
        entropy,
        sorted_actions[:, -1] - sorted_actions[:, -2],
    ])


def fit_logistic(probabilities: np.ndarray, truth: np.ndarray, regularization: float) -> dict:
    features = gate_features(probabilities)
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale < 1e-6] = 1.0
    x = (features - mean) / scale
    y = (truth != 5).astype(np.float64)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        weights, bias = parameters[:-1], parameters[-1]
        logits = x @ weights + bias
        loss = np.mean(np.logaddexp(0.0, logits) - y * logits) + 0.5 * regularization * np.sum(weights ** 2)
        probabilities_fg = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        gradient_weights = x.T @ (probabilities_fg - y) / len(y) + regularization * weights
        gradient_bias = np.mean(probabilities_fg - y)
        return float(loss), np.concatenate([gradient_weights, [gradient_bias]])

    initial = np.zeros(x.shape[1] + 1, dtype=np.float64)
    result = minimize(objective, initial, method="L-BFGS-B", jac=True, options={"maxiter": 500})
    if not result.success:
        raise RuntimeError(f"logistic optimization failed: {result.message}")
    return {
        "regularization": regularization,
        "mean": mean,
        "scale": scale,
        "weights": result.x[:-1],
        "bias": float(result.x[-1]),
    }


def gate_probability(model: dict, probabilities: np.ndarray) -> np.ndarray:
    x = (gate_features(probabilities) - model["mean"]) / model["scale"]
    logits = x @ model["weights"] + model["bias"]
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))


def binary_log_loss(truth: np.ndarray, probability: np.ndarray) -> float:
    y = (truth != 5).astype(np.float64)
    p = np.clip(probability, 1e-8, 1 - 1e-8)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def choose_regularization(probabilities: np.ndarray, truth: np.ndarray, videos: np.ndarray) -> float:
    candidates = (0.001, 0.01, 0.1, 1.0)
    best = None
    for regularization in candidates:
        losses = []
        for video in sorted(set(videos.tolist())):
            validation = videos == video
            training = ~validation
            model = fit_logistic(probabilities[training], truth[training], regularization)
            losses.append(binary_log_loss(truth[validation], gate_probability(model, probabilities[validation])))
        value = float(np.mean(losses))
        key = (-value, -abs(math.log10(regularization) + 1))
        if best is None or key > best[0]:
            best = (key, regularization, value)
    assert best is not None
    return float(best[1])


def fit_two_stage(probabilities: np.ndarray, truth: np.ndarray, videos: np.ndarray) -> dict:
    foreground_thresholds = fit_foreground_thresholds(probabilities, truth)
    regularization = choose_regularization(probabilities, truth, videos)
    gate_model = fit_logistic(probabilities, truth, regularization)
    fg_probability = gate_probability(gate_model, probabilities)
    action_prediction = decode_foreground(probabilities, foreground_thresholds)
    best = None
    for threshold in np.arange(0.05, 0.951, 0.025):
        prediction = action_prediction.copy()
        prediction[fg_probability < threshold] = 5
        score = fast_metrics(truth, prediction)
        key = (score["six_class_macro_f1"], score["action_macro_f1"], score["accuracy"],
               score["background_recall"], -abs(float(threshold) - 0.5))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    assert best is not None
    return {
        "foreground_thresholds": foreground_thresholds,
        "gate_model": gate_model,
        "gate_threshold": best[1],
    }


def decode_two_stage(model: dict, probabilities: np.ndarray) -> np.ndarray:
    prediction = decode_foreground(probabilities, model["foreground_thresholds"])
    prediction[gate_probability(model["gate_model"], probabilities) < model["gate_threshold"]] = 5
    return prediction


def serializable_parameters(method: str, model) -> dict:
    if method in {"legacy_thresholds", "multivideo_action_thresholds", "multivideo_balanced_thresholds"}:
        return {label: float(model[index]) for index, label in enumerate(LABELS)}
    if method == "probability_bg_gate":
        gate = model["gate_model"]
        return {
            "foreground_thresholds": {label: float(model["foreground_thresholds"][index])
                                      for index, label in enumerate(ACTION_LABELS)},
            "gate_threshold": float(model["gate_threshold"]),
            "gate_features": ["log_p_bg", "log_max_action_p", "entropy", "top_action_margin"],
            "regularization": float(gate["regularization"]),
            "feature_mean": gate["mean"].tolist(),
            "feature_scale": gate["scale"].tolist(),
            "weights": gate["weights"].tolist(),
            "bias": float(gate["bias"]),
        }
    return {}


def fit_method(method: str, probabilities: np.ndarray, truth: np.ndarray, videos: np.ndarray):
    if method == "raw_argmax":
        return None
    if method == "legacy_thresholds":
        return threshold_array(LEGACY_THRESHOLDS)
    if method == "multivideo_action_thresholds":
        return fit_thresholds(probabilities, truth, "action_first")
    if method == "multivideo_balanced_thresholds":
        return fit_thresholds(probabilities, truth, "balanced")
    if method == "probability_bg_gate":
        return fit_two_stage(probabilities, truth, videos)
    raise ValueError(method)


def decode_method(method: str, model, probabilities: np.ndarray) -> np.ndarray:
    if method == "raw_argmax":
        return np.argmax(probabilities, axis=1)
    if method in {"legacy_thresholds", "multivideo_action_thresholds", "multivideo_balanced_thresholds"}:
        return decode_threshold(probabilities, model)
    if method == "probability_bg_gate":
        return decode_two_stage(model, probabilities)
    raise ValueError(method)


def temporal_summary(rows: list[dict]) -> dict:
    manifest = json.loads(QUERY_MANIFEST.read_text(encoding="utf-8"))
    by_video = {video["video_id"]: [row for row in rows if row["video_id"] == video["video_id"]]
                for video in manifest["videos"]}
    output = {}
    for name, remove_bg in (("with_background", False), ("without_background", True)):
        values = [temporal_metrics(video_rows, remove_bg) for video_rows in by_video.values()]
        aggregate = {key: sum(item[key] for item in values) / len(values)
                     for key in ("Acc", "Edit", "F1@10", "F1@25", "F1@50")}
        aggregate["FEA"] = sum(aggregate.values()) / 5
        output[name] = aggregate
    return output


def main() -> None:
    development_rows = load_development_rows()
    development_probabilities, development_truth, development_videos = arrays(development_rows)
    groups = sorted({row["session_group"] for row in development_rows})
    if len(groups) != 2:
        raise SystemExit(f"expected two groups, got {groups}")
    methods = (
        "raw_argmax",
        "legacy_thresholds",
        "multivideo_action_thresholds",
        "multivideo_balanced_thresholds",
        "probability_bg_gate",
    )
    folds = []
    cross_validated_predictions = {method: np.full(len(development_rows), -1, dtype=np.int64) for method in methods}
    for train_group, validation_group in ((groups[0], groups[1]), (groups[1], groups[0])):
        training = np.asarray([row["session_group"] == train_group for row in development_rows])
        validation = np.asarray([row["session_group"] == validation_group for row in development_rows])
        fold = {"train_group": train_group, "validation_group": validation_group, "methods": {}}
        for method in methods:
            model = fit_method(method, development_probabilities[training], development_truth[training],
                               development_videos[training])
            prediction = decode_method(method, model, development_probabilities[validation])
            cross_validated_predictions[method][validation] = prediction
            validation_rows = [row for row, keep in zip(development_rows, validation) if keep]
            fold["methods"][method] = {
                "parameters": serializable_parameters(method, model),
                "validation": full_metrics(validation_rows, prediction),
            }
        folds.append(fold)
    if any(np.any(prediction < 0) for prediction in cross_validated_predictions.values()):
        raise RuntimeError("cross-validation did not predict every development row")

    cross_validation = {
        method: full_metrics(development_rows, prediction)
        for method, prediction in cross_validated_predictions.items()
    }
    ranking = sorted(methods, key=lambda method: (
        cross_validation[method]["six_class_macro_f1"],
        cross_validation[method]["action_macro_f1"],
        cross_validation[method]["accuracy"],
        cross_validation[method]["background_recall"],
    ), reverse=True)
    selected_method = ranking[0]

    query_rows = json.loads(QUERY.read_text(encoding="utf-8"))
    query_probabilities, query_truth, _ = arrays(query_rows)
    query_evaluation = {}
    fitted_parameters = {}
    selected_query_rows = None
    for method in methods:
        model = fit_method(method, development_probabilities, development_truth, development_videos)
        prediction = decode_method(method, model, query_probabilities)
        fitted_parameters[method] = serializable_parameters(method, model)
        query_evaluation[method] = full_metrics(query_rows, prediction)
        if method == selected_method:
            selected_query_rows = [
                {**row, "raw_prediction": row["prediction"], "prediction": INDEX_TO_LABEL[int(value)],
                 "confidence": row["probabilities"][INDEX_TO_LABEL[int(value)]]}
                for row, value in zip(query_rows, prediction)
            ]
    assert selected_query_rows is not None
    selected_temporal = temporal_summary(selected_query_rows)

    output = {
        "schema_version": "jev-fpc-probability-alignment-v1",
        "status": "complete",
        "protocol": {
            "model": "StartLux-Decision-v1.1-4B-AutoThink",
            "think_mode": "never",
            "prompt_layout": "visual_current",
            "reference_view": "original automatic early|middle|late sheets",
            "development_videos": len(set(development_videos.tolist())),
            "development_windows": len(development_rows),
            "cross_validation": "recording-session-group A->B and B->A; every development row predicted out of group",
            "selection_metric": "pooled out-of-group six-class Macro-F1, then action Macro-F1, accuracy, BG recall",
            "query_gt_used_for_method_or_parameter_selection": False,
            "query_inference_reused": str(QUERY),
        },
        "ranking": ranking,
        "selected_method": selected_method,
        "folds": folds,
        "cross_validation": cross_validation,
        "fitted_on_all_development": fitted_parameters,
        "query_diagnostics_after_lock": query_evaluation,
        "selected_query_temporal_macro_over_videos": selected_temporal,
    }
    (EXP / "evaluation.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (EXP / "selected_query_predictions.json").write_text(
        json.dumps(selected_query_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    def pct(value: float) -> str:
        return f"{100 * value:.2f}%"

    cv_lines = []
    query_lines = []
    display = {
        "raw_argmax": "raw argmax",
        "legacy_thresholds": "原单循环阈值",
        "multivideo_action_thresholds": "多视频动作优先阈值",
        "multivideo_balanced_thresholds": "多视频六类平衡阈值",
        "probability_bg_gate": "概率BG gate + 前景对齐",
    }
    for method in methods:
        cv = cross_validation[method]
        query = query_evaluation[method]
        mark = "（CV选中）" if method == selected_method else ""
        cv_lines.append(
            f"| {display[method]}{mark} | {pct(cv['accuracy'])} | {pct(cv['action_macro_f1'])} | "
            f"{pct(cv['six_class_macro_f1'])} | {pct(cv['background_recall'])} |"
        )
        query_lines.append(
            f"| {display[method]}{mark} | {pct(query['accuracy'])} | {pct(query['action_macro_f1'])} | "
            f"{pct(query['six_class_macro_f1'])} | {pct(query['background_recall'])} |"
        )
    temporal = selected_temporal
    report = f"""# 小米主FPC：多视频概率对齐实验

## 协议

- 参考图、模型、prompt和采样均冻结为原`visual_current`基线；本实验只改变六类概率的解码；
- 开发集为12条support-split视频、{len(development_rows)}窗，参考视频和6条Query均排除；
- 两组按录制时段互斥，执行A→B、B→A双向验证，每个开发窗口都只由未见过其录制组的参数预测；
- 方法按开发集out-of-group六类Macro-F1锁定，之后才应用到既有4,786条Query概率；
- Query GT不参与方法、阈值、正则或gate选择，Query结果只是锁定后的测试诊断。

## 双向跨时段验证（275窗）

| 方法 | Accuracy | 动作Macro-F1 | 六类Macro-F1 | BG Recall |
|---|---:|---:|---:|---:|
{chr(10).join(cv_lines)}

锁定方法：`{selected_method}`。

## 冻结Query诊断（4,786窗）

| 方法 | Accuracy | 动作Macro-F1 | 六类Macro-F1 | BG Recall |
|---|---:|---:|---:|---:|
{chr(10).join(query_lines)}

## 锁定方法的标准时序指标（六视频宏平均）

| 口径 | Acc | Edit | F1@10 | F1@25 | F1@50 | FEA |
|---|---:|---:|---:|---:|---:|---:|
| 含BG | {temporal['with_background']['Acc']:.2f} | {temporal['with_background']['Edit']:.2f} | {temporal['with_background']['F1@10']:.2f} | {temporal['with_background']['F1@25']:.2f} | {temporal['with_background']['F1@50']:.2f} | {temporal['with_background']['FEA']:.2f} |
| 去BG | {temporal['without_background']['Acc']:.2f} | {temporal['without_background']['Edit']:.2f} | {temporal['without_background']['F1@10']:.2f} | {temporal['without_background']['F1@25']:.2f} | {temporal['without_background']['F1@50']:.2f} | {temporal['without_background']['FEA']:.2f} |
"""
    (EXP / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "selected_method": selected_method,
        "ranking": ranking,
        "cross_validation": {
            method: {key: cross_validation[method][key]
                     for key in ("accuracy", "action_macro_f1", "six_class_macro_f1", "background_recall")}
            for method in methods
        },
        "query": {
            method: {key: query_evaluation[method][key]
                     for key in ("accuracy", "action_macro_f1", "six_class_macro_f1", "background_recall")}
            for method in methods
        },
        "report": str(EXP / "REPORT.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
