#!/usr/bin/env python3
"""Evaluate the soft cyclic prior on the locked dual-reference probability ensemble."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import evaluate_probability_alignment as pa
import evaluate_probability_ensemble as pe
import evaluate_soft_temporal_prior as st


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "probability_alignment_v1" / "reference_ensemble" / "temporal_prior_v1"
QUERY_SELECTED = ROOT / "probability_alignment_v1" / "reference_ensemble" / "selected_query_predictions.json"


def fit_balanced(probabilities: np.ndarray, truth: np.ndarray, videos: np.ndarray) -> np.ndarray:
    del videos
    return pa.fit_thresholds(probabilities, truth, "balanced")


def decode_balanced(thresholds: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    return pa.decode_threshold(probabilities, thresholds)


def threshold_emissions(thresholds: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    adjusted_log = np.log(clipped / thresholds[None, :])
    background = np.maximum(adjusted_log[:, 5], 0.0)
    return np.concatenate([
        adjusted_log[:, :5],
        np.repeat(background[:, None], 5, axis=1),
    ], axis=1)


def compact(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in ("accuracy", "action_macro_f1", "six_class_macro_f1", "background_recall")
    }


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    rows = pa.load_development_rows()
    original, truth, videos = pa.arrays(rows)
    alternate = pe.alternate_probabilities(rows)
    probabilities = pe.ensemble("arithmetic_mean", original, alternate)
    groups = sorted({row["session_group"] for row in rows})

    base_cv = np.full(len(rows), -1, dtype=np.int64)
    temporal_cv = np.full(len(rows), -1, dtype=np.int64)
    folds = []
    for train_group, validation_group in ((groups[0], groups[1]), (groups[1], groups[0])):
        training = np.asarray([row["session_group"] == train_group for row in rows])
        validation = np.asarray([row["session_group"] == validation_group for row in rows])
        train_indices = np.flatnonzero(training)
        train_rows = [rows[index] for index in train_indices]
        parameter, inner = st.select_parameters_leave_one_video_out(
            train_rows,
            probabilities[training],
            truth[training],
            videos[training],
            f"ensemble outer {train_group}",
            fit_function=fit_balanced,
            emission_function=threshold_emissions,
            decode_function=decode_balanced,
        )
        print(f"[ensemble outer] {train_group} -> {validation_group}; selected {parameter}", flush=True)
        model = fit_balanced(probabilities[training], truth[training], videos[training])
        validation_indices = np.flatnonzero(validation)
        base_cv[validation_indices] = decode_balanced(model, probabilities[validation])
        decoded = st.decode_videos(
            model,
            probabilities,
            rows,
            videos,
            parameter,
            validation,
            emission_function=threshold_emissions,
        )
        temporal_cv[validation_indices] = decoded[validation_indices]
        validation_rows = [rows[index] for index in validation_indices]
        folds.append({
            "train_group": train_group,
            "validation_group": validation_group,
            "selected_parameters": {"change_penalty": parameter[0], "order_penalty": parameter[1]},
            "inner_leave_one_video_out": inner,
            "base_validation": compact(st.detailed_metrics(validation_rows, base_cv[validation_indices])),
            "temporal_validation": compact(st.detailed_metrics(validation_rows, decoded[validation_indices])),
        })
    if np.any(base_cv < 0) or np.any(temporal_cv < 0):
        raise RuntimeError("outer cross-validation did not cover all development rows")
    development_base = st.detailed_metrics(rows, base_cv)
    development_temporal = st.detailed_metrics(rows, temporal_cv)

    final_parameter, final_selection = st.select_parameters_leave_one_video_out(
        rows,
        probabilities,
        truth,
        videos,
        "ensemble final support LOO",
        fit_function=fit_balanced,
        emission_function=threshold_emissions,
        decode_function=decode_balanced,
    )
    print(f"[ensemble final] support-selected parameter {final_parameter}", flush=True)
    final_thresholds = fit_balanced(probabilities, truth, videos)

    query_source = json.loads(QUERY_SELECTED.read_text(encoding="utf-8"))
    query_rows = [
        {**row, "probabilities": row["ensemble_probabilities"]}
        for row in query_source
    ]
    query_probabilities, _, query_videos = pa.arrays(query_rows)
    query_base_prediction = decode_balanced(final_thresholds, query_probabilities)
    expected_base = np.asarray([pa.LABEL_TO_INDEX[row["prediction"]] for row in query_source], dtype=np.int64)
    if not np.array_equal(query_base_prediction, expected_base):
        raise RuntimeError("recomputed ensemble baseline does not match locked query predictions")
    query_temporal_prediction = st.decode_videos(
        final_thresholds,
        query_probabilities,
        query_rows,
        query_videos,
        final_parameter,
        emission_function=threshold_emissions,
    )
    query_base_rows = st.with_predictions(query_rows, query_base_prediction)
    query_temporal_rows = st.with_predictions(query_rows, query_temporal_prediction, query_base_prediction)
    query_base = st.detailed_metrics(query_rows, query_base_prediction)
    query_temporal = st.detailed_metrics(query_rows, query_temporal_prediction)
    base_temporal = st.temporal_macro(query_base_rows)
    soft_temporal = st.temporal_macro(query_temporal_rows)
    changed = sum(row["changed_by_temporal_prior"] for row in query_temporal_rows)
    fixed = sum(row["changed_by_temporal_prior"] and row["temporal_correct"] for row in query_temporal_rows)
    broken = sum(row["changed_by_temporal_prior"] and row["base_gate_correct"] and not row["temporal_correct"]
                 for row in query_temporal_rows)

    evaluation = {
        "schema_version": "jev-fpc-ensemble-soft-temporal-prior-v1",
        "status": "complete",
        "protocol": {
            "base": "fixed 1:1 arithmetic mean of original and three-video reference probabilities + balanced thresholds",
            "temporal_model": "10-state cyclic Viterbi with stage-carrying BG states",
            "outer_validation": "recording-session-group A->B and B->A",
            "outer_parameter_selection": "leave-one-video-out inside each outer training group",
            "final_parameter_selection": "leave-one-video-out over all 12 support development videos",
            "query_gt_used_for_selection": False,
        },
        "outer_folds": folds,
        "development_outer_cross_validation": {
            "base": compact(development_base),
            "soft_temporal_prior": compact(development_temporal),
        },
        "final_support_selection": final_selection,
        "fitted_on_all_support": {
            "thresholds": {label: float(final_thresholds[index]) for index, label in enumerate(pa.LABELS)},
            "temporal": {"change_penalty": final_parameter[0], "order_penalty": final_parameter[1]},
        },
        "query_diagnostics_after_lock": {
            "base": compact(query_base),
            "soft_temporal_prior": compact(query_temporal),
            "base_temporal": base_temporal,
            "soft_temporal": soft_temporal,
            "changed_windows": changed,
            "changed_to_correct": fixed,
            "changed_correct_to_wrong": broken,
            "net_correct": fixed - broken,
            "base_confusion": st.confusion(query_base_rows),
            "soft_temporal_confusion": st.confusion(query_temporal_rows),
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
        "正向：软时序先验在跨时段开发验证与冻结Query上均继续改善双参考组集成。"
        if cv_delta > 0 and query_delta > 0
        else "未形成稳定正向结果：软时序先验没有在开发验证与冻结Query上同时改善双参考组集成。"
    )
    report = f"""# 双参考组集成 + 软时序先验

## 协议

- 基线为已锁定的双参考组1:1算术平均概率与六类平衡阈值；
- 时序模型、参数网格与原参考实验相同；
- Viterbi读取整条视频概率序列，属于离线双向解码；
- 外层按录制时段A→B、B→A验证，内层留一视频选择参数；
- 最终参数只由12条support开发视频锁定，Query GT不参与选择。

## 双向跨时段外层验证（275窗）

| 方法 | Accuracy | 动作Macro-F1 | 六类Macro-F1 | BG Recall |
|---|---:|---:|---:|---:|
| 双参考组集成 | {st.pct(development_base['accuracy'])} | {st.pct(development_base['action_macro_f1'])} | {st.pct(development_base['six_class_macro_f1'])} | {st.pct(development_base['background_recall'])} |
| + 软时序先验 | {st.pct(development_temporal['accuracy'])} | {st.pct(development_temporal['action_macro_f1'])} | {st.pct(development_temporal['six_class_macro_f1'])} | {st.pct(development_temporal['background_recall'])} |

最终support侧锁定：`change_penalty={final_parameter[0]:.2f}`，`order_penalty={final_parameter[1]:.2f}`。

## 冻结Query诊断（4,786窗）

| 方法 | Accuracy | 动作Macro-F1 | 六类Macro-F1 | BG Recall |
|---|---:|---:|---:|---:|
| 双参考组集成 | {st.pct(query_base['accuracy'])} | {st.pct(query_base['action_macro_f1'])} | {st.pct(query_base['six_class_macro_f1'])} | {st.pct(query_base['background_recall'])} |
| + 软时序先验 | {st.pct(query_temporal['accuracy'])} | {st.pct(query_temporal['action_macro_f1'])} | {st.pct(query_temporal['six_class_macro_f1'])} | {st.pct(query_temporal['background_recall'])} |

时序先验改变{changed}个窗口，其中{fixed}个由错变对、{broken}个由对变错，净增加{fixed-broken}个正确窗口。

## Query标准时序指标（六视频宏平均）

| 方法/口径 | Acc | Edit | F1@10 | F1@25 | F1@50 | FEA |
|---|---:|---:|---:|---:|---:|---:|
{st.temporal_row('双参考，含BG', base_temporal['with_background'])}
{st.temporal_row('+软时序，含BG', soft_temporal['with_background'])}
{st.temporal_row('双参考，去BG', base_temporal['without_background'])}
{st.temporal_row('+软时序，去BG', soft_temporal['without_background'])}

## 结论

{verdict}

限制：逐视频和逐类别收益并不完全一致，顺序先验不能替代视觉证据；同时该方法是离线双向解码，不是因果在线解码。
"""
    (EXP / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "outer_cv": {"base": compact(development_base), "temporal": compact(development_temporal)},
        "final_parameter": {"change_penalty": final_parameter[0], "order_penalty": final_parameter[1]},
        "query": {
            "base": compact(query_base),
            "temporal": compact(query_temporal),
            "changed": changed,
            "fixed": fixed,
            "broken": broken,
            "net": fixed - broken,
            "base_temporal": base_temporal,
            "soft_temporal": soft_temporal,
        },
        "report": str(EXP / "REPORT.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
