"""Portable cached replay, calibration and four-reference GPU inference."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from numerics import (
    LABELS, PARAMETERS, classification_metrics, decode_threshold, fast_metrics,
    fit_thresholds, score_key, temporal_metrics, threshold_emissions, viterbi,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
BANKS = ("V1", "V2", "V3", "V6")


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(split):
    return read(DATA / "inputs" / f"{split}.json")["windows"]


def outputs(split, cache):
    values = {}
    for path in sorted((cache / split).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            if record["id"] in values:
                raise ValueError(f"Duplicate response {record['id']}")
            values[record["id"]] = record
    return values


def cube(split, cache):
    records = outputs(split, cache)
    expected = {}
    values = []
    for row in rows(split):
        probabilities = []
        for bank in BANKS:
            key = f"{split}:{row['video_id']}:{row['window_id']}:{bank}"
            record = records[key]
            assert record["input_sha256"] == row["request_sha256"][bank]
            p = np.asarray(record["probabilities"])
            assert p.shape == (6,) and np.isfinite(p).all() and (p >= 0).all() and abs(p.sum() - 1) < 1e-6
            expected[key] = True
            probabilities.append(p)
        values.append(probabilities)
    assert set(records) == set(expected)
    return np.asarray(values)


def decode(windows, probabilities, thresholds, penalties):
    prediction = np.full(len(windows), -1, dtype=int)
    groups = defaultdict(list)
    for i, row in enumerate(windows):
        groups[row["video_id"]].append(i)
    for indices in groups.values():
        indices.sort(key=lambda i: windows[i]["window_id"])
        # Split decoding at any gap in the evaluation grid.
        boundaries = [0] + [j for j in range(1, len(indices))
                            if windows[indices[j]]["window_id"] != windows[indices[j - 1]]["window_id"] + 1]
        boundaries.append(len(indices))
        for start, stop in zip(boundaries, boundaries[1:]):
            selected = indices[start:stop]
            prediction[selected] = viterbi(threshold_emissions(np.asarray(thresholds), probabilities[selected]), *penalties)
    assert (prediction >= 0).all()
    return prediction


def score(windows, prediction, truth):
    data = [dict(video_id=r["video_id"], window_id=r["window_id"], gt=LABELS[int(t)], prediction=LABELS[int(p)])
            for r, p, t in zip(windows, prediction, truth, strict=True)]
    classification = classification_metrics(data)
    classification["foreground_recall"] = float(np.sum((truth != 5) & (prediction != 5)) / np.sum(truth != 5))
    cm = np.zeros((6, 6), dtype=int)
    np.add.at(cm, (truth, prediction), 1)
    classification["confusion"] = {gt: {p: int(cm[i, j]) for j, p in enumerate(LABELS)} for i, gt in enumerate(LABELS)}
    groups = defaultdict(list)
    for row in data:
        groups[row["video_id"]].append(row)
    temporal = {}
    for name, remove_bg in (("with_background", False), ("without_background", True)):
        per_video = [temporal_metrics(items, remove_bg) for items in groups.values()]
        temporal[name] = {key: float(np.mean([item[key] for item in per_video])) for key in per_video[0]}
    return dict(classification=classification, temporal=temporal)


def replay(args):
    windows = rows("query")
    probabilities = cube("query", args.cache)
    lock = read(ROOT / "config/LOCK.json")
    expected = read(DATA / "results/four_worker_full_query_v1/query_predictions.json")
    reference = read(DATA / "results/four_worker_full_query_v1/evaluation.json")
    audit = read(DATA / "results/four_worker_full_query_v1/data_audit.json")
    truth = np.array([LABELS.index(label) for label in read(DATA / "evaluation_labels/query.json")["labels"]])
    assert expected["keys"] == [[r["video_id"], r["window_id"]] for r in windows]
    result = {}
    for arm, config in lock["arms"].items():
        p = probabilities[:, [BANKS.index(bank) for bank in config["banks"]]].mean(axis=1)
        variants = {
            f"{arm}_raw": p.argmax(axis=1),
            f"{arm}_threshold": decode_threshold(p, np.asarray(config["thresholds"])),
            f"{arm}_temporal": decode(windows, p, config["thresholds"], config["penalties"]),
        }
        for name, prediction in variants.items():
            assert prediction.tolist() == expected["predictions"][name], name
            metrics = score(windows, prediction, truth)
            original = reference["query"][name]
            for key in ("accuracy", "action_macro_f1", "foreground_recall", "background_recall"):
                assert np.isclose(metrics["classification"][key], original["classification"][key]), (name, key)
            assert metrics["classification"]["confusion"] == original["classification"]["confusion"]
            for policy, values in metrics["temporal"].items():
                for key, value in values.items():
                    assert np.isclose(value, original["temporal"][policy][key]), (name, policy, key)
            result[name] = metrics
    v4 = np.array([row["video_id"] == audit["v4_video"] for row in windows])
    main = np.asarray(expected["predictions"]["four_temporal"])
    result["V4_recording_isolated"] = score([row for row, keep in zip(windows, v4) if keep], main[v4], truth[v4])
    report = dict(status="passed", query_windows=len(windows), exact_prediction_replay_arms=len(expected["predictions"]), results=result)
    save(args.output / "REPLAY_VALIDATION.json", report)
    print(json.dumps(dict(status="passed", query_windows=len(windows), main=result["four_temporal"]["classification"]), indent=2))


def calibrate(args):
    windows = rows("support")
    probabilities = cube("support", args.cache)
    truth = np.array([LABELS.index(label) for label in read(DATA / "evaluation_labels/support.json")["labels"]])
    threshold_mask = np.array([r["block"] == "threshold" for r in windows])
    temporal_rows = [r for r, use in zip(windows, threshold_mask) if not use]
    config = read(ROOT / "config/LOCK.json")
    fitted = {}
    for name, arm in config["arms"].items():
        p = probabilities[:, [BANKS.index(bank) for bank in arm["banks"]]].mean(axis=1)
        thresholds = fit_thresholds(p[threshold_mask], truth[threshold_mask], "balanced")
        scores = {pair: fast_metrics(truth[~threshold_mask], decode(temporal_rows, p[~threshold_mask], thresholds, pair))
                  for pair in PARAMETERS}
        pair = max(scores, key=lambda item: score_key(scores[item], item))
        assert thresholds.tolist() == arm["thresholds"] and list(pair) == arm["penalties"], name
        fitted[name] = dict(thresholds=thresholds.tolist(), penalties=list(pair), selection_score=scores[pair])
        print(name, fitted[name], flush=True)
    save(args.output / "CALIBRATION_REPLAY.json", dict(status="passed", arms=fitted))


def build_sheets(args):
    from PIL import Image
    for row in rows(args.split):
        destination = args.sheets / row["sheet"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas = Image.new("RGB", (1928, 384), "white")
        for i, sample in enumerate(row["samples"]):
            source = args.frames / row["source_video_id"] / f"{sample['frame_index_1based']:06d}.jpg"
            with Image.open(source) as frame:
                image = frame.convert("RGB")
            if image.size != (384, 384):
                image = image.resize((384, 384), Image.Resampling.LANCZOS)
            canvas.paste(image, (i * 384 + (8 if i >= 2 else 0), 0))
        canvas.save(destination, quality=90, subsampling=0)
        assert sha(destination) == row["sheet_sha256"], destination
    print(f"Built and hash-verified {len(rows(args.split))} {args.split} sheets")


def infer(args):
    from PIL import Image
    from vllm import LLM, SamplingParams

    sys.path.insert(0, str(ROOT / "vendor"))
    from startlux_decide.prompt import check_tokenizer, from_systemone, messages, plain_prompt
    from startlux_decide.images import place_images

    prompt = read(ROOT / "config/PROMPT.json")
    refs = read(DATA / "config/REFERENCES.json")
    for bank in BANKS:
        assert [sha(DATA / "references" / bank / f"{label}.jpg") for label in LABELS] == refs[bank]
    requests = [(row, bank) for row in rows(args.split) for bank in BANKS][args.rank::args.world]
    output = args.output / args.split / f"rank{args.rank}.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    prior = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    done = {r["id"]: r for r in prior}
    assert len(done) == len(prior)
    expected = {f"{args.split}:{row['video_id']}:{row['window_id']}:{bank}": row["request_sha256"][bank] for row, bank in requests}
    assert set(done) <= set(expected)
    assert all(r["input_sha256"] == expected[key] for key, r in done.items())
    llm = LLM(model=str(args.model), dtype="bfloat16", tensor_parallel_size=1, max_model_len=8192,
              gpu_memory_utilization=.88, limit_mm_per_prompt={"image": 8},
              mm_processor_kwargs={"min_pixels": 65536, "max_pixels": 1048576},
              generation_config="vllm", logprobs_mode="processed_logprobs", max_num_seqs=8, seed=42)
    ids = check_tokenizer(llm.get_tokenizer())[:6]
    sampling = SamplingParams(max_tokens=1, temperature=1, logprobs=6, allowed_token_ids=ids)
    rendered = plain_prompt(llm.get_tokenizer(), messages(from_systemone(
        place_images(prompt["state"], 7), prompt["questions"]["action"])))
    for start in range(0, len(requests), 8):
        batch = [(r, bank) for r, bank in requests[start:start + 8]
                 if f"{args.split}:{r['video_id']}:{r['window_id']}:{bank}" not in done]
        if not batch:
            continue
        prompts = []
        for row, bank in batch:
            query = args.sheets / row["sheet"]
            assert sha(query) == row["sheet_sha256"]
            paths = [query] + [DATA / "references" / bank / f"{label}.jpg" for label in LABELS]
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            prompts.append({"prompt": rendered, "multi_modal_data": {"image": images}})
        began = time.monotonic()
        generated = llm.generate(prompts, sampling, use_tqdm=False)
        elapsed = (time.monotonic() - began) / len(batch)
        with output.open("a") as stream:
            for (row, bank), result in zip(batch, generated, strict=True):
                logits = np.array([result.outputs[0].logprobs[0][i].logprob for i in ids]) / .9276
                p = np.exp(logits - logits.max())
                p /= p.sum()
                key = f"{args.split}:{row['video_id']}:{row['window_id']}:{bank}"
                record = dict(id=key, input_sha256=row["request_sha256"][bank], probabilities=p.tolist(),
                              elapsed_s=elapsed, input_tokens=len(result.prompt_token_ids), batch_size=len(batch))
                stream.write(json.dumps(record) + "\n")
                done[key] = record
            stream.flush()
        print(len(done), len(requests), flush=True)


def main():
    global DATA
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA, help="Local private data bundle")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("replay", replay), ("calibrate", calibrate)):
        command = sub.add_parser(name)
        command.add_argument("--cache", type=Path, default=None)
        command.add_argument("--output", type=Path, default=ROOT / "output")
        command.set_defaults(handler=handler)
    command = sub.add_parser("build-sheets")
    command.add_argument("--split", required=True, choices=("support", "query"))
    command.add_argument("--frames", required=True, type=Path)
    command.add_argument("--sheets", type=Path, default=ROOT / "sheets")
    command.set_defaults(handler=build_sheets)
    command = sub.add_parser("infer")
    command.add_argument("--split", required=True, choices=("support", "query"))
    command.add_argument("--model", required=True, type=Path)
    command.add_argument("--sheets", type=Path, default=ROOT / "sheets")
    command.add_argument("--output", type=Path, default=ROOT / "output/inference")
    command.add_argument("--rank", type=int, default=0)
    command.add_argument("--world", type=int, default=1)
    command.set_defaults(handler=infer)
    args = parser.parse_args()
    DATA = args.data.resolve()
    if args.command in ("replay", "calibrate") and args.cache is None:
        args.cache = DATA / "cache"
    if args.command == "infer" and not 0 <= args.rank < args.world:
        parser.error("require 0 <= rank < world")
    args.handler(args)


if __name__ == "__main__":
    main()
