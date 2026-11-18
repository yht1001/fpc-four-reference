#!/usr/bin/env python3
"""Run resumable StartLux/JEV inference on frozen main-FPC query sheets."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import mimetypes
import time
from datetime import datetime, timezone
from pathlib import Path

import aiohttp


OUT = Path(__file__).resolve().parents[1]
REFERENCE_ROOT = OUT / "support" / "references"
CARD_PATH = OUT / "config" / "action_card_drafts.json"
OPTION_IDS = ("A01", "A02", "A03", "A04", "A05", "BG")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def build_request(
    row: dict,
    summaries: dict[str, str],
    think_mode: str,
    arm: str,
    reference_root: Path = REFERENCE_ROOT,
) -> dict:
    images = [data_uri(Path(row["sheet"]))]
    state = [
        "Query evidence is one five-panel temporal sheet: <image>",
        "The first two panels are low-rate context from the preceding two seconds.",
        "The last three panels densely sample the target interval: the most recent 0.5 seconds.",
        "Classify only the last three target panels; use the first two panels only as context.",
        "Candidate visual references; every sheet is ordered early | middle | late:",
    ]
    for option in OPTION_IDS:
        state.append(f"{option}: <image>")
        images.append(data_uri(reference_root / f"{option}.jpg"))
    return {
        "state": "\n".join(state),
        "questions": {
            "action": {
                "type": "choice",
                "instructions": (
                    "Which candidate best matches the action occurring in the target interval, the most recent "
                    "0.5 seconds represented by the last three query panels? The first two panels are earlier "
                    "context only. Compare the target with the candidate references and option descriptions."
                ),
                "criteria": {
                    option: (
                        summaries[option]
                        if arm == "card_current"
                        else f"The action in the last three query panels matches candidate {option}."
                    )
                    for option in OPTION_IDS
                },
            }
        },
        "images": images,
        "think_mode": think_mode,
        "none_check": False,
    }


def request_for_sampling(row: dict, summaries: dict[str, str], think_mode: str, arm: str,
                         reference_root: Path) -> dict:
    """Keep the baseline prompt exact; update panel counts for denser sheets."""
    request = build_request(row, summaries, think_mode, arm, reference_root)
    target_count = int(row.get("target_panel_count", 3))
    if target_count == 3:
        return request
    if target_count != 5:
        raise ValueError("supported target panel counts are 3 and 5")
    request["state"] = request["state"].replace("five-panel", "seven-panel").replace("last three", "last five")
    question = request["questions"]["action"]
    question["instructions"] = question["instructions"].replace("last three", "last five")
    question["criteria"] = {k: v.replace("last three", "last five") for k, v in question["criteria"].items()}
    return request


def completed_keys(path: Path, think_mode: str, arm: str) -> set[tuple[str, int]]:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if (row.get("status") == "ok" and row.get("think_mode") == think_mode
                and row.get("prompt_layout") == arm):
            done.add((str(row["video_id"]), int(row["window_id"])))
    return done


async def main_async(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cards = json.loads(CARD_PATH.read_text(encoding="utf-8"))["cards"]
    summaries = {option: cards[option]["option_summary_en"] for option in OPTION_IDS}
    rows = [row for video in manifest["videos"] for row in video["windows"]]
    if args.limit:
        rows = rows[:args.limit]
    done = completed_keys(args.output, args.think_mode, args.arm) if args.resume else set()
    pending = [row for row in rows if (row["video_id"], int(row["window_id"])) not in done]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "schema_version": "jev-fpc-inference-run-v1",
        "model": "StartLux-Decision-v1.1-4B-AutoThink",
        "think_mode": args.think_mode,
        "prompt_layout": args.arm,
        "option_summaries_used": args.arm == "card_current",
        "decision_rule": "raw argmax over A01-A05 and BG; no query-GT thresholding",
        "query_manifest": str(args.manifest),
        "query_manifest_sha256": digest(args.manifest),
        "action_cards": str(CARD_PATH.resolve()),
        "action_cards_sha256": digest(CARD_PATH),
        "reference_root": str(args.reference_root),
        "reference_sha256": {option: digest(args.reference_root / f"{option}.jpg") for option in OPTION_IDS},
        "total_rows": len(rows),
        "sampling": manifest.get("sampling"),
        "endpoint": args.url,
    }
    run_manifest_path = args.output.parent / "run_manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"total={len(rows)} completed={len(done)} pending={len(pending)}", flush=True)
    if not pending:
        return

    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()
    async with aiohttp.ClientSession(timeout=timeout, connector=connector, trust_env=False) as session:
        async def run_one(row: dict) -> None:
            async with semaphore:
                payload = request_for_sampling(row, summaries, args.think_mode, args.arm, args.reference_root)
                last_error = None
                for attempt in range(1, args.retries + 1):
                    started = time.monotonic()
                    try:
                        async with session.post(args.url, json=payload) as response:
                            body = await response.text()
                            if response.status != 200:
                                raise RuntimeError(f"HTTP {response.status}: {body[:1000]}")
                            answer = json.loads(body)
                        record = {
                            "schema_version": "jev-fpc-query-result-v1",
                            "status": "ok",
                            "model": run_manifest["model"],
                            "think_mode": args.think_mode,
                            "prompt_layout": args.arm,
                            "video_id": row["video_id"],
                            "window_id": int(row["window_id"]),
                            "target_start_s": row["target_start_s"],
                            "target_end_s": row["target_end_s"],
                            "target_center_s": row["target_center_s"],
                            "query_sheet": row["sheet"],
                            "elapsed_s": round(time.monotonic() - started, 6),
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                            "response": answer,
                        }
                        async with write_lock:
                            with args.output.open("a", encoding="utf-8") as handle:
                                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                                handle.flush()
                        choice = answer["answers"]["action"]["choice"]
                        probability = answer["answers"]["action"]["probabilities"][choice]
                        if int(row["window_id"]) % args.log_every == 0:
                            print(
                                f"[{row['video_id']} {int(row['window_id']):05d}] "
                                f"{choice} p={probability:.4f} {record['elapsed_s']:.2f}s",
                                flush=True,
                            )
                        return
                    except Exception as exc:
                        last_error = f"{type(exc).__name__}: {exc}"
                        print(
                            f"[{row['video_id']} {int(row['window_id']):05d}] "
                            f"attempt {attempt}/{args.retries}: {last_error}",
                            flush=True,
                        )
                        if attempt < args.retries:
                            await asyncio.sleep(min(2 ** attempt, 10))
                async with write_lock:
                    with args.output.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({
                            "schema_version": "jev-fpc-query-result-v1",
                            "status": "error",
                            "video_id": row["video_id"],
                            "window_id": int(row["window_id"]),
                            "think_mode": args.think_mode,
                            "error": last_error,
                            "completed_at": datetime.now(timezone.utc).isoformat(),
                        }, ensure_ascii=False) + "\n")
                        handle.flush()

        await asyncio.gather(*(run_one(row) for row in pending))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=OUT / "query" / "manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--arm", choices=("card_current", "visual_current"), default="card_current")
    parser.add_argument("--url", default="http://127.0.0.1:8091/v1/systemone")
    parser.add_argument("--think-mode", choices=("never", "auto", "always"), default="never")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if args.output is None:
        dirname = "results" if args.arm == "card_current" else f"results_{args.arm}"
        args.output = OUT / dirname / "raw_predictions.jsonl"
    args.manifest = args.manifest.resolve()
    args.output = args.output.resolve()
    args.reference_root = args.reference_root.resolve()
    missing_references = [str(args.reference_root / f"{option}.jpg") for option in OPTION_IDS
                          if not (args.reference_root / f"{option}.jpg").is_file()]
    if missing_references:
        raise SystemExit(f"missing reference sheets: {missing_references}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
