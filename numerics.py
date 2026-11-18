"""Numerical routines extracted unchanged from the archived experiment sources."""

from __future__ import annotations

from collections import Counter

import itertools

import numpy as np

LABELS = ("A01", "A02", "A03", "A04", "A05", "BG")

ACTION_LABELS = LABELS[:-1]

PARAMETERS = tuple((float(c), float(o)) for c in (0,.1,.25,.5,.75,1,1.5) for o in (0,.1,.25,.5,.75,1,1.5,2,3))

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

def score_key(score: dict, parameter: tuple[float, float]) -> tuple:
    return (
        score["six_class_macro_f1"],
        score["action_macro_f1"],
        score["accuracy"],
        score["background_recall"],
        -sum(parameter),
    )

def threshold_emissions(thresholds: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    adjusted_log = np.log(clipped / thresholds[None, :])
    background = np.maximum(adjusted_log[:, 5], 0.0)
    return np.concatenate([
        adjusted_log[:, :5],
        np.repeat(background[:, None], 5, axis=1),
    ], axis=1)
