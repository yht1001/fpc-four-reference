#!/usr/bin/env python3
"""Cross-validate fixed probability ensembles from two independent reference panels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import evaluate_probability_alignment as core


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "probability_alignment_v1" / "reference_ensemble"
ALT_RAW = ROOT / "probability_alignment_v1" / "multivideo_references" / "raw_predictions.jsonl"
METHODS = (
    "raw_argmax",
    "legacy_thresholds",
    "multivideo_action_thresholds",
    "multivideo_balanced_thresholds",
    "probability_bg_gate",
)
REPRESENTATIONS = ("arithmetic_mean", "geometric_mean")


def alternate_probabilities(rows: list[dict]) -> np.ndarray:
    results = core.latest_results(ALT_RAW)
    missing = [(row["video_id"], row["window_id"]) for row in rows
               if (row["video_id"], row["window_id"]) not in results]
    if missing:
        raise SystemExit(f"alternate predictions missing: {len(missing)}")
    return np.asarray([
        [results[(row["video_id"], row["window_id"])]["response"]["answers"]["action"]
         ["probabilities"][label] for label in core.LABELS]
        for row in rows
    ], dtype=np.float64)


def ensemble(name: str, original: np.ndarray, alternate: np.ndarray) -> np.ndarray:
    if name == "arithmetic_mean":
        values = (original + alternate) / 2
    elif name == "geometric_mean":
        values = np.sqrt(np.clip(original, 1e-12, 1) * np.clip(alternate, 1e-12, 1))
    else:
        raise ValueError(name)
    return values / values.sum(axis=1, keepdims=True)


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    rows = core.load_development_rows()
    original, truth, videos = core.arrays(rows)
    alternate = alternate_probabilities(rows)
    groups = sorted({row["session_group"] for row in rows})
    candidates = [(representation, method) for representation in REPRESENTATIONS for method in METHODS]
    predictions = {candidate: np.full(len(rows), -1, dtype=np.int64) for candidate in candidates}
    folds = []

    for train_group, validation_group in ((groups[0], groups[1]), (groups[1], groups[0])):
        training = np.asarray([row["session_group"] == train_group for row in rows])
        validation = np.asarray([row["session_group"] == validation_group for row in rows])
        validation_rows = [row for row, keep in zip(rows, validation) if keep]
        fold = {"train_group": train_group, "validation_group": validation_group, "candidates": {}}
        for representation in REPRESENTATIONS:
            probability = ensemble(representation, original, alternate)
            for method in METHODS:
                candidate = (representation, method)
                model = core.fit_method(method, probability[training], truth[training], videos[training])
                prediction = core.decode_method(method, model, probability[validation])
                predictions[candidate][validation] = prediction
                fold["candidates"]["/".join(candidate)] = {
                    "parameters": core.serializable_parameters(method, model),
                    "validation": core.full_metrics(validation_rows, prediction),
                }
        folds.append(fold)

    metrics = {candidate: core.full_metrics(rows, prediction)
               for candidate, prediction in predictions.items()}
    ranking = sorted(candidates, key=lambda candidate: (
        metrics[candidate]["six_class_macro_f1"],
        metrics[candidate]["action_macro_f1"],
        metrics[candidate]["accuracy"],
        metrics[candidate]["background_recall"],
    ), reverse=True)
    selected = ranking[0]
    output = {
        "schema_version": "jev-fpc-reference-probability-ensemble-v1",
        "status": "complete",
        "protocol": {
            "representations": list(REPRESENTATIONS),
            "ensemble_weight": "fixed 0.5/0.5; not fitted",
            "development_videos": len(set(videos.tolist())),
            "development_windows": len(rows),
            "cross_validation": "recording-session-group A->B and B->A",
            "query_gt_used": False,
            "query_inference_run": False,
        },
        "ranking": ["/".join(candidate) for candidate in ranking],
        "selected_candidate": "/".join(selected),
        "folds": folds,
        "cross_validation": {"/".join(candidate): value for candidate, value in metrics.items()},
    }
    (EXP / "evaluation.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = []
    for candidate in candidates:
        value = metrics[candidate]
        mark = "（CV选中）" if candidate == selected else ""
        lines.append(
            f"| {' / '.join(candidate)}{mark} | {100*value['accuracy']:.2f}% | "
            f"{100*value['action_macro_f1']:.2f}% | {100*value['six_class_macro_f1']:.2f}% | "
            f"{100*value['background_recall']:.2f}% |"
        )
    (EXP / "REPORT.md").write_text(f"""# 双参考组概率集成验证

- 对原单视频参考组与三视频多样化参考组的六类概率做固定1:1算术均值或几何均值；
- 仍在同一12视频、275窗上执行录制时段A→B、B→A隔离验证；
- 集成权重固定，未用开发集或Query拟合；Query尚未运行。

| 候选 | Accuracy | 动作Macro-F1 | 六类Macro-F1 | BG Recall |
|---|---:|---:|---:|---:|
{chr(10).join(lines)}
""", encoding="utf-8")
    print(json.dumps({
        "selected_candidate": "/".join(selected),
        "ranking": ["/".join(candidate) for candidate in ranking],
        "cross_validation": {
            "/".join(candidate): {key: metrics[candidate][key] for key in (
                "accuracy", "action_macro_f1", "six_class_macro_f1", "background_recall"
            )} for candidate in candidates
        },
        "report": str(EXP / "REPORT.md"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
