"""Run the frozen protocol, waiting for free cards and keeping resumable shards."""

import fcntl
import os
import signal
import subprocess
import sys
import time

from study import ROOT, b

ENV = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="2",
           TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
DEADLINE = time.monotonic() + 24 * 3600


def progress(state, **extra):
    counts = {}
    for stage in ("query",):
        counts[stage] = sum(
            b.read(path)["completed"] for path in ROOT.glob(f"progress/{stage}_rank*.json")
        )
    b.save(ROOT / "STATUS.json", {
        "state": state, "pid": os.getpid(), "completed_requests": counts,
        "expected_requests": {"query": b.read(ROOT / "RUN_LOCK.json")["new_requests"]}, **extra,
    })


def idle_cards():
    cards = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,memory.free,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True)
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits",
    ], text=True)
    occupied = {line.split(",")[0].strip() for line in apps.splitlines() if "," in line}
    return {
        uuid for uuid, free, util in (map(str.strip, line.split(",")) for line in cards.splitlines())
        if int(free) >= 22000 and int(util) <= 5 and uuid not in occupied
    }


def infer(split, world):
    pending = {
        rank for rank in range(world)
        if not (ROOT / "progress" / f"complete_{split}_rank{rank}.json").is_file()
    }
    active = {}
    previous = set()
    try:
        while pending or active:
            if time.monotonic() > DEADLINE:
                raise TimeoutError("24-hour run window elapsed")
            for rank, (card, worker) in list(active.items()):
                code = worker.poll()
                if code is None:
                    continue
                del active[rank]
                if code:
                    log_path = ROOT / "logs" / f"{split}_{rank}.log"
                    tail = log_path.read_text(errors="replace")[-16000:]
                    if "Free memory on device" in tail and "desired GPU memory utilization" in tail:
                        pending.add(rank)
                        previous.discard(card)
                        continue
                    raise RuntimeError(f"{split} shard {rank} exited with code {code}. See {log_path}")
                if not (ROOT / "progress" / f"complete_{split}_rank{rank}.json").is_file():
                    raise RuntimeError(f"{split} shard {rank} exited without a completion marker")
            if pending:
                free = idle_cards() - {card for card, _ in active.values()}
                for card in sorted(free & previous):
                    if not pending:
                        break
                    rank = min(pending)
                    pending.remove(rank)
                    with (ROOT / "logs" / f"{split}_{rank}.log").open("a") as stream:
                        worker = subprocess.Popen(
                            [str(b.VLLM_PY), str(ROOT / "study.py"), split, str(rank), str(world)],
                            env=dict(ENV, CUDA_VISIBLE_DEVICES=card), stdout=stream, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True,
                        )
                    active[rank] = (card, worker)
                previous = free
            progress(f"inference_{split}" if active else f"waiting_for_GPU_{split}",
                     pending_ranks=sorted(pending), active={rank: card for rank, (card, _) in active.items()})
            if pending or active:
                time.sleep(10)
    finally:
        for _, worker in active.values():
            if worker.poll() is None:
                os.killpg(worker.pid, signal.SIGTERM)
        for _, worker in active.values():
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(worker.pid, signal.SIGKILL)
                worker.wait()


def cpu(script, *args):
    subprocess.run([str(b.PY), str(ROOT / script), *args], env=ENV, check=True)


def main():
    progress("checking_inputs")
    lock = b.read(ROOT / "RUN_LOCK.json")
    b.check(lock)
    b.check({"source_sha256": lock["weights_sha256"]})
    infer("query", 8)
    if not (ROOT / "evaluation.json").is_file():
        progress("scoring_Query")
        cpu("study.py", "evaluate")
    progress("validating")
    cpu("study.py", "validate")
    progress("complete")


if __name__ == "__main__":
    (ROOT / "logs").mkdir(exist_ok=True)
    with (ROOT / ".run.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            main()
        except BaseException as exc:
            progress("failed", error=str(exc))
            raise
