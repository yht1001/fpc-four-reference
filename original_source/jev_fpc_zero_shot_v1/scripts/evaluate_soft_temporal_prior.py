#!/usr/bin/env python3
"""Evaluate a cyclic soft temporal prior on top of the locked probability BG gate.

All hyperparameters are selected with support-side labels only.  The outer estimate
uses recording-session-group A->B/B->A validation; each outer training group uses
leave-one-video-out selection for temporal hyperparameters.  Query GT is read only
after the final support-side model and hyperparameters have been locked.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import evaluate_probability_alignment as pa
from evaluate_query import LABELS, classification_metrics, temporal_metrics


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "probability_alignment_v1" / "temporal_prior_v1"
QUERY = ROOT / "results_visual_current" / "predictions.json"
ACTION_LABELS = LABELS[:-1]
INDEX_TO_LABEL = dict(enumerate(LABELS))
PARAMETERS = tuple(
    (float(change_penalty), float(order_penalty))
    for change_penalty in (0.0, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50)
    for order_penalty in (0.0, 0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00)
)


def logit(value: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(value, 1e-8, 1.0 - 1e-8)
    return np.log(clipped) - np.log1p(-clipped)


def temporal_emissions(model: dict, probabilities: np.ndarray) -> np.ndarray:
    """Return emissions for A01..A05 plus five stage-carrying BG states.

    At zero transition penalties, framewise argmax exactly reproduces the locked
    two-stage probability gate.  BG states share the same visible BG label but
    remember the most recent action stage while the video is in background.
    """
    fg_probability = pa.gate_probability(model["gate_model"], probabilities)
    gate_margin = logit(fg_probability) - float(logit(model["gate_threshold"]))
    adjusted_actions = np.clip(
        probabilities[:, :5] / model["foreground_thresholds"][None, :], 1e-12, None
    )
    relative_action_log = np.log(adjusted_actions) - np.log(
        np.max(adjusted_actions, axis=1, keepdims=True)
    )
    action_emissions = gate_margin[:, None] + relative_action_log
    background_emissions = np.zeros((len(probabilities), 5), dtype=np.float64)
    return np.concatenate([action_emissions, background_emissions], axis=1)


def unexpected_cost(previous_action: int, next_action: int) -> int:
    distance = (next_action - previous_action) % 5
    return {0: 1, 1: 0, 2: 1, 3: 2, 4: 3}[distance]


def transition_matrix(change_penalty: float, order_penalty: float) -> np.ndarray:
    """Build a soft cyclic transition matrix over 5 action + 5 BG-stage states."""
    matrix = np.full((10, 10), -np.inf, dtype=np.float64)
    for action in range(5):
        bg_state = 5 + action
        matrix[action, action] = 0.0
        matrix[action, bg_state] = -change_penalty
        matrix[bg_state, bg_state] = 0.0
        for next_action in range(5):
            if next_action == action:
                continue
            penalty = change_penalty + order_penalty * unexpected_cost(action, next_action)
            matrix[action, next_action] = -penalty
        for next_action in range(5):
            penalty = change_penalty + order_penalty * unexpected_cost(action, next_action)
            matrix[bg_state, next_action] = -penalty
    return matrix


def viterbi(emissions: np.ndarray, change_penalty: float, order_penalty: float) -> np.ndarray:
    transitions = transition_matrix(change_penalty, order_penalty)
    scores = emissions[0].copy()
    backpointers = np.zeros((len(emissions), emissions.shape[1]), dtype=np.int16)
    for time_index in range(1, len(emissions)):
        candidates = scores[:, None] + transitions
        backpointers[time_index] = np.argmax(candidates, axis=0)
        scores = candidates[backpointers[time_index], np.arange(emissions.shape[1])] + emissions[time_index]
    states = np.zeros(len(emissions), dtype=np.int64)
    states[-1] = int(np.argmax(scores))
    for time_index in range(len(emissions) - 1, 0, -1):
        states[time_index - 1] = backpointers[time_index, states[time_index]]
    return np.where(states < 5, states, 5)


def video_indices(rows: list[dict], videos: np.ndarray, selected: np.ndarray | None = None) -> list[np.ndarray]:
    mask = np.ones(len(rows), dtype=bool) if selected is None else selected
    output = []
    for video_id in sorted(set(videos[mask].tolist())):
        indices = np.flatnonzero(mask & (videos == video_id))
        indices = indices[np.argsort([int(rows[index]["window_id"]) for index in indices])]
        output.append(indices)
    return output


def decode_videos(
    model: dict,
    probabilities: np.ndarray,
    rows: list[dict],
    videos: np.ndarray,
    parameter: tuple[float, float],
    selected: np.ndarray | None = None,
    emission_function=temporal_emissions,
) -> np.ndarray:
    prediction = np.full(len(rows), -1, dtype=np.int64)
    for indices in video_indices(rows, videos, selected):
        emissions = emission_function(model, probabilities[indices])
        prediction[indices] = viterbi(emissions, *parameter)
    return prediction


def score_key(score: dict, parameter: tuple[float, float]) -> tuple:
    return (
        score["six_class_macro_f1"],
        score["action_macro_f1"],
        score["accuracy"],
        score["background_recall"],
        -sum(parameter),
    )


def select_parameters_leave_one_video_out(
    rows: list[dict],
    probabilities: np.ndarray,
    truth: np.ndarray,
    videos: np.ndarray,
    label: str,
    fit_function=pa.fit_two_stage,
    emission_function=temporal_emissions,
    decode_function=pa.decode_two_stage,
) -> tuple[tuple[float, float], dict]:
    predictions = {parameter: np.full(len(rows), -1, dtype=np.int64) for parameter in PARAMETERS}
    unique_videos = sorted(set(videos.tolist()))
    for number, held_out in enumerate(unique_videos, start=1):
        print(f"[{label}] fit {number}/{len(unique_videos)}; held out {held_out}", flush=True)
        validation = videos == held_out
        training = ~validation
        model = fit_function(probabilities[training], truth[training], videos[training])
        indices = np.flatnonzero(validation)
        indices = indices[np.argsort([int(rows[index]["window_id"]) for index in indices])]
        emissions = emission_function(model, probabilities[indices])
        base = decode_function(model, probabilities[indices])
        no_temporal = viterbi(emissions, 0.0, 0.0)
        if not np.array_equal(base, no_temporal):
            raise RuntimeError("zero-penalty temporal decoder does not reproduce probability gate")
        for parameter in PARAMETERS:
            predictions[parameter][indices] = viterbi(emissions, *parameter)
    if any(np.any(value < 0) for value in predictions.values()):
        raise RuntimeError("leave-one-video-out selection did not predict every row")
    scores = {parameter: pa.fast_metrics(truth, prediction) for parameter, prediction in predictions.items()}
    ranking = sorted(PARAMETERS, key=lambda parameter: score_key(scores[parameter], parameter), reverse=True)
    selected = ranking[0]
    top = [
        {
            "change_penalty": parameter[0],
            "order_penalty": parameter[1],
            **scores[parameter],
        }
        for parameter in ranking[:10]
    ]
    return selected, {"selected": selected, "selected_metrics": scores[selected], "top_candidates": top}


def with_predictions(rows: list[dict], prediction: np.ndarray, base_prediction: np.ndarray | None = None) -> list[dict]:
    output = []
    for index, (row, value) in enumerate(zip(rows, prediction)):
        label = INDEX_TO_LABEL[int(value)]
        item = {**row, "prediction": label, "confidence": float(row["probabilities"][label])}
        if base_prediction is not None:
            base_label = INDEX_TO_LABEL[int(base_prediction[index])]
            item.update({
                "base_gate_prediction": base_label,
                "changed_by_temporal_prior": base_label != label,
                "base_gate_correct": base_label == row["gt"],
                "temporal_correct": label == row["gt"],
            })
        output.append(item)
    return output


def detailed_metrics(rows: list[dict], prediction: np.ndarray) -> dict:
    decoded = with_predictions(rows, prediction)
    result = classification_metrics(decoded)
    result["six_class_macro_f1"] = sum(result["per_class"][label]["f1"] for label in LABELS) / len(LABELS)
    return result


def temporal_macro(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["video_id"]].append(row)
    for values in grouped.values():
        values.sort(key=lambda row: int(row["window_id"]))
    output = {}
    for name, remove_bg in (("with_background", False), ("without_background", True)):
        values = [temporal_metrics(video_rows, remove_bg) for video_rows in grouped.values()]
        aggregate = {
            key: sum(item[key] for item in values) / len(values)
            for key in ("Acc", "Edit", "F1@10", "F1@25", "F1@50")
        }
        aggregate["FEA"] = sum(aggregate.values()) / 5
        output[name] = aggregate
    return output


def confusion(rows: list[dict]) -> dict:
    return {
        gt: {pred: sum(row["gt"] == gt and row["prediction"] == pred for row in rows) for pred in LABELS}
        for gt in LABELS
    }


def compact(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in ("accuracy", "action_macro_f1", "six_class_macro_f1", "background_recall")
    }


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def temporal_row(name: str, value: dict) -> str:
    return (
        f"| {name} | {value['Acc']:.2f} | {value['Edit']:.2f} | {value['F1@10']:.2f} | "
        f"{value['F1@25']:.2f} | {value['F1@50']:.2f} | {value['FEA']:.2f} |"
    )


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    development_rows = pa.load_development_rows()
    development_probabilities, development_truth, development_videos = pa.arrays(development_rows)
    groups = sorted({row["session_group"] for row in development_rows})
    if len(groups) != 2:
        raise RuntimeError(f"expected two session groups, got {groups}")

    base_cross_validated = np.full(len(development_rows), -1, dtype=np.int64)
    temporal_cross_validated = np.full(len(development_rows), -1, dtype=np.int64)
    outer_folds = []
    for train_group, validation_group in ((groups[0], groups[1]), (groups[1], groups[0])):
        training = np.asarray([row["session_group"] == train_group for row in development_rows])
        validation = np.asarray([row["session_group"] == validation_group for row in development_rows])
        train_indices = np.flatnonzero(training)
        train_rows = [development_rows[index] for index in train_indices]
        parameter, inner = select_parameters_leave_one_video_out(
            train_rows,
            development_probabilities[training],
            development_truth[training],
            development_videos[training],
            f"outer {train_group}",
        )
        print(f"[outer] {train_group} -> {validation_group}; selected {parameter}", flush=True)
        model = pa.fit_two_stage(
            development_probabilities[training], development_truth[training], development_videos[training]
        )
        base_prediction = pa.decode_two_stage(model, development_probabilities[validation])
        validation_indices = np.flatnonzero(validation)
        base_cross_validated[validation_indices] = base_prediction
        decoded = decode_videos(
            model,
            development_probabilities,
            development_rows,
            development_videos,
            parameter,
            validation,
        )
        temporal_cross_validated[validation_indices] = decoded[validation_indices]
        validation_rows = [development_rows[index] for index in validation_indices]
        outer_folds.append({
            "train_group": train_group,
            "validation_group": validation_group,
            "selected_parameters": {"change_penalty": parameter[0], "order_penalty": parameter[1]},
            "inner_leave_one_video_out": inner,
            "base_gate_validation": compact(detailed_metrics(validation_rows, base_prediction)),
            "temporal_validation": compact(detailed_metrics(validation_rows, decoded[validation_indices])),
        })
    if np.any(base_cross_validated < 0) or np.any(temporal_cross_validated < 0):
        raise RuntimeError("outer cross-validation did not predict every development row")

    development_base_rows = with_predictions(development_rows, base_cross_validated)
    development_temporal_rows = with_predictions(development_rows, temporal_cross_validated)
    development_base = detailed_metrics(development_rows, base_cross_validated)
    development_temporal = detailed_metrics(development_rows, temporal_cross_validated)

    final_parameter, final_selection = select_parameters_leave_one_video_out(
        development_rows,
        development_probabilities,
        development_truth,
        development_videos,
        "final support LOO",
    )
    print(f"[final] support-selected parameter {final_parameter}", flush=True)
    final_model = pa.fit_two_stage(development_probabilities, development_truth, development_videos)

    query_rows = json.loads(QUERY.read_text(encoding="utf-8"))
    query_probabilities, _, query_videos = pa.arrays(query_rows)
    query_base_prediction = pa.decode_two_stage(final_model, query_probabilities)
    query_temporal_prediction = decode_videos(
        final_model, query_probabilities, query_rows, query_videos, final_parameter
    )
    query_base_rows = with_predictions(query_rows, query_base_prediction)
    query_temporal_rows = with_predictions(query_rows, query_temporal_prediction, query_base_prediction)
    query_base = detailed_metrics(query_rows, query_base_prediction)
    query_temporal = detailed_metrics(query_rows, query_temporal_prediction)
    query_base_temporal = temporal_macro(query_base_rows)
    query_temporal_temporal = temporal_macro(query_temporal_rows)

    changed = sum(row["changed_by_temporal_prior"] for row in query_temporal_rows)
    fixed = sum(row["changed_by_temporal_prior"] and row["temporal_correct"] for row in query_temporal_rows)
    broken = sum(row["changed_by_temporal_prior"] and row["base_gate_correct"] and not row["temporal_correct"]
                 for row in query_temporal_rows)
    net = fixed - broken

    evaluation = {
        "schema_version": "jev-fpc-soft-temporal-prior-v1",
        "status": "complete",
        "protocol": {
            "base": "original-reference probability BG gate + foreground alignment",
            "temporal_model": "10-state cyclic Viterbi: A01-A05 plus five BG states carrying last action stage",
            "expected_order": "A01->A02->A03->A04->A05->A01; BG may appear between actions",
            "hyperparameter_grid": {
                "change_penalty": sorted({value[0] for value in PARAMETERS}),
                "order_penalty": sorted({value[1] for value in PARAMETERS}),
            },
            "outer_validation": "recording-session-group A->B and B->A",
            "outer_parameter_selection": "leave-one-video-out inside the outer training group",
            "final_parameter_selection": "leave-one-video-out across all 12 support development videos",
            "query_gt_used_for_selection": False,
            "query_model_inference_reused": str(QUERY),
        },
        "outer_folds": outer_folds,
        "development_outer_cross_validation": {
            "base_gate": compact(development_base),
            "soft_temporal_prior": compact(development_temporal),
            "base_temporal": temporal_macro(development_base_rows),
            "soft_temporal": temporal_macro(development_temporal_rows),
        },
        "final_support_selection": final_selection,
        "fitted_on_all_support": {
            "probability_gate": pa.serializable_parameters("probability_bg_gate", final_model),
            "temporal": {"change_penalty": final_parameter[0], "order_penalty": final_parameter[1]},
        },
        "query_diagnostics_after_lock": {
            "base_gate": compact(query_base),
            "soft_temporal_prior": compact(query_temporal),
            "base_temporal": query_base_temporal,
            "soft_temporal": query_temporal_temporal,
            "changed_windows": changed,
            "changed_to_correct": fixed,
            "changed_correct_to_wrong": broken,
            "net_correct": net,
            "base_confusion": confusion(query_base_rows),
            "soft_temporal_confusion": confusion(query_temporal_rows),
            "base_per_class": query_base["per_class"],
            "soft_temporal_per_class": query_temporal["per_class"],
        },
    }
    (EXP / "evaluation.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (EXP / "selected_query_predictions.json").write_text(
        json.dumps(query_temporal_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    cv_delta = development_temporal["six_class_macro_f1"] - development_base["six_class_macro_f1"]
    query_delta = query_temporal["six_class_macro_f1"] - query_base["six_class_macro_f1"]
    verdict = (
        "正向：跨时段开发验证和冻结Query的六类Macro-F1均提升。"
        if cv_delta > 0 and query_delta > 0
        else "未形成稳定正向结果：开发验证与冻结Query没有同时提升。"
    )
    report = f"""# 小米主FPC：软时序先验实验

## 协议

- 基线固定为“原参考 + 概率BG gate”，不重新运行JEV，只复用现有六类概率；
- 时序模型为10状态循环Viterbi：5个动作状态和5个携带上一动作阶段的BG状态；
- 期望顺序为`A01→A02→A03→A04→A05→A01`，允许动作间出现BG，也允许付出惩罚后跳步或回退；
- Viterbi读取整条视频的概率序列，属于离线双向解码，不等同于实时在线检测；
- 外层按录制时段A→B、B→A验证；每个外层训练组内部再留一视频选择时序参数；
- 最终参数只用12条support开发视频留一验证锁定，6条Query GT不参与参数或方法选择。

## 双向跨时段外层验证（275窗）

| 方法 | Accuracy | 动作Macro-F1 | 六类Macro-F1 | BG Recall |
|---|---:|---:|---:|---:|
| 原参考 + 概率BG gate | {pct(development_base['accuracy'])} | {pct(development_base['action_macro_f1'])} | {pct(development_base['six_class_macro_f1'])} | {pct(development_base['background_recall'])} |
| + 软时序先验 | {pct(development_temporal['accuracy'])} | {pct(development_temporal['action_macro_f1'])} | {pct(development_temporal['six_class_macro_f1'])} | {pct(development_temporal['background_recall'])} |

两个外层训练组分别选择：`{outer_folds[0]['selected_parameters']}`和`{outer_folds[1]['selected_parameters']}`。

## Support侧最终锁定

- `change_penalty = {final_parameter[0]:.2f}`：通用切换/持续时间惩罚；0表示support没有选择额外平滑；
- `order_penalty = {final_parameter[1]:.2f}`：惩罚偏离循环顺序的跳转；
- 选择依据：12条support视频留一预测的六类Macro-F1优先，其次动作Macro-F1、Accuracy和BG Recall。

## 冻结Query诊断（4,786窗）

| 方法 | Accuracy | 动作Macro-F1 | 六类Macro-F1 | BG Recall |
|---|---:|---:|---:|---:|
| 原参考 + 概率BG gate | {pct(query_base['accuracy'])} | {pct(query_base['action_macro_f1'])} | {pct(query_base['six_class_macro_f1'])} | {pct(query_base['background_recall'])} |
| + 软时序先验 | {pct(query_temporal['accuracy'])} | {pct(query_temporal['action_macro_f1'])} | {pct(query_temporal['six_class_macro_f1'])} | {pct(query_temporal['background_recall'])} |

时序先验改变{changed}个窗口，其中{fixed}个由错变对、{broken}个由对变错，净增加{net}个正确窗口。

## Query标准时序指标（六视频宏平均）

| 方法/口径 | Acc | Edit | F1@10 | F1@25 | F1@50 | FEA |
|---|---:|---:|---:|---:|---:|---:|
{temporal_row('基线，含BG', query_base_temporal['with_background'])}
{temporal_row('软时序，含BG', query_temporal_temporal['with_background'])}
{temporal_row('基线，去BG', query_base_temporal['without_background'])}
{temporal_row('软时序，去BG', query_temporal_temporal['without_background'])}

## 结论

{verdict}

限制：逐视频收益并不完全一致，顺序先验不能修复视觉概率本身严重错误的视频；同时该方法使用整条视频的未来概率，不是因果在线解码。
"""
    (EXP / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "outer_cv": {
            "base": compact(development_base),
            "temporal": compact(development_temporal),
        },
        "final_parameter": {"change_penalty": final_parameter[0], "order_penalty": final_parameter[1]},
        "query": {
            "base": compact(query_base),
            "temporal": compact(query_temporal),
            "changed": changed,
            "fixed": fixed,
            "broken": broken,
            "net": net,
            "base_temporal": query_base_temporal,
            "soft_temporal": query_temporal_temporal,
        },
        "report": str(EXP / "REPORT.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
