"""Master controller for the split-pipeline job matrix.

Maintains an explicit TODO list (state JSON) and polls every 10 minutes:
  1. Scan results dirs for partial pickles (truth of "done").
  2. Query squeue to identify in-flight preorder tasks.
  3. For each cell (dataset, noise, algorithm), compute the missing
     (repeat, fold) tasks and submit a new array covering as many as
     the free SLURM slots allow (cap = 25 preorder tasks per user).
  4. When all 4 algorithms of a (dataset, noise) cell are complete
     (100/100 partials), submit a merge+evaluate job once.
  5. Report a human-readable progress table to the log.

The cap math: def QOS MaxSubmit=40, observed empirically as a hard
per-user limit including the user's 15 GPU LLM jobs. So preorder
tasks must stay ≤ 25.

Auto-fix:
  * Tasks whose array job left the queue without producing a pickle
    are eligible for resubmission on the next poll.
  * QOSMaxSubmit errors abort the current submission only; the loop
    retries on the next poll.

Usage:
    nohup python scripts/ablations/master_controller.py \\
        > slurm_logs/master_controller.out 2>&1 &
    # Stop: kill <pid>  (in-flight SLURM jobs are NOT cancelled)
"""

from __future__ import annotations

import datetime as _dt
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = Path("/tmp/preorder_master_state.json")
LOG_PATH = ROOT / "slurm_logs" / "master_controller.log"
REPORT_PATH = ROOT / "slurm_logs" / "master_controller_report.txt"

POLL_SECS = int(os.environ.get("POLL_SECS", "600"))   # 10 min default
PREORDER_CAP = 36     # max preorder tasks in queue (40 def-QOS cap, 4 buffer for extras + retry)
SLURM_USER = os.environ["USER"]

DATASETS = {
    "enron": {
        "results_dir": "results/full_enron_split",
        "mem": "256G",
        "time": "2-00:00:00",
    },
}
NOISES = ["0.0", "0.1", "0.2", "0.3"]
ALGOS = ["bopos", "clr", "br", "cc"]
N_REPEATS = 5
N_FOLDS = 5
TASKS_PER_CELL_ALGO = N_REPEATS * N_FOLDS   # 25


def log(msg: str) -> None:
    line = f"[{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_FILE.exists():
        with STATE_FILE.open() as f:
            return json.load(f)
    return {"arrays": {}}   # jid → {dataset, noise, algorithm, task_ids}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_FILE)


def _suffix_for(algo: str) -> str:
    return "" if algo == "bopos" else f"_{algo}"


def partials_on_disk(results_dir: str, dataset: str, noise: str, algo: str) -> set[tuple[int, int]]:
    """Returns set of (repeat, fold) that have a partial pickle on disk."""
    suffix = _suffix_for(algo)
    pattern = str(
        ROOT / results_dir
        / f"dataset_{dataset}_noisy_{noise}{suffix}_r*_f*.pkl"
    )
    rx = re.compile(rf".*_r(\d+)_f(\d+)\.pkl$")
    done: set[tuple[int, int]] = set()
    for p in glob.glob(pattern):
        m = rx.match(p)
        if m:
            done.add((int(m.group(1)), int(m.group(2))))
    return done


def merged_on_disk(results_dir: str, dataset: str, noise: str) -> bool:
    """True if all four consolidated pickles (bopos+clr+br+cc) and an
    evaluation_*_dataset_level.csv exist for this cell."""
    base = ROOT / results_dir
    for algo in ALGOS:
        sfx = _suffix_for(algo)
        if not (base / f"dataset_{dataset}_noisy_{noise}{sfx}.pkl").exists():
            return False
    return (base / f"evaluation_{dataset}_noisy_{noise}_dataset_level.csv").exists()


def squeue_preorder_jids() -> set[str]:
    """Returns set of array job IDs (master id only) currently in queue for
    preorder-* jobs."""
    try:
        out = subprocess.check_output(
            ["squeue", "-u", SLURM_USER, "-h", "-o", "%i %j"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return set()
    jids: set[str] = set()
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        jid_part, name = parts
        if not name.startswith("preorder-"):
            continue
        master = jid_part.split("_", 1)[0]
        jids.add(master)
    return jids


def squeue_preorder_task_count() -> int:
    try:
        out = subprocess.check_output(
            ["squeue", "-u", SLURM_USER, "-r", "-h", "-o", "%j"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        return 0
    return sum(1 for line in out.splitlines() if line.startswith("preorder-"))


def compact_ranges(ids: list[int]) -> str:
    """[0,1,2,5,7,8] → '0-2,5,7-8'."""
    if not ids:
        return ""
    ids = sorted(set(ids))
    out = []
    s = e = ids[0]
    for n in ids[1:]:
        if n == e + 1:
            e = n
        else:
            out.append(f"{s}-{e}" if s != e else f"{s}")
            s = e = n
    out.append(f"{s}-{e}" if s != e else f"{s}")
    return ",".join(out)


def task_id_of(repeat: int, fold: int) -> int:
    return repeat * N_FOLDS + fold


def rf_of(task_id: int) -> tuple[int, int]:
    return divmod(task_id, N_FOLDS)


def submit_split_array(
    dataset: str, noise: str, algorithm: str, task_ids: list[int],
    results_dir: str, mem: str, time_limit: str,
) -> str | None:
    """Returns SLURM array job id, or None on failure."""
    rng = compact_ranges(task_ids)
    env = (
        f"ALL,DATASET={dataset},NOISE_RATE={noise},RESULTS_DIR={results_dir},"
        f"ALGORITHM={algorithm},BASE_LEARNER=RF,SOLVER=highs"
    )
    cmd = [
        "sbatch", "--parsable",
        f"--array={rng}",
        f"--time={time_limit}",
        "--cpus-per-task=4",
        f"--mem={mem}",
        f"--export={env}",
        str(ROOT / "scripts/ablations/full_train_split.sbatch"),
    ]
    try:
        out = subprocess.check_output(
            cmd, text=True, stderr=subprocess.STDOUT, cwd=ROOT,
        ).strip()
        return out
    except subprocess.CalledProcessError as e:
        log(f"  [submit error] {dataset} n={noise} {algorithm} array={rng}: "
            f"{e.output.strip().splitlines()[-1] if e.output else 'unknown'}")
        return None


def submit_merge(dataset: str, noise: str, results_dir: str) -> str | None:
    cmd = [
        "sbatch", "--parsable",
        f"--export=ALL,DATASET={dataset},NOISE_RATE={noise},RESULTS_DIR={results_dir}",
        str(ROOT / "scripts/ablations/merge_and_eval.sbatch"),
    ]
    try:
        out = subprocess.check_output(
            cmd, text=True, stderr=subprocess.STDOUT, cwd=ROOT,
        ).strip()
        return out
    except subprocess.CalledProcessError as e:
        log(f"  [merge submit error] {dataset} n={noise}: "
            f"{e.output.strip().splitlines()[-1] if e.output else 'unknown'}")
        return None


def write_report(progress: dict, free_slots: int, in_flight_jids: set[str]) -> None:
    lines = []
    lines.append(f"# master controller report   {_dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"# preorder cap=25, free_slots={free_slots}, in_flight_arrays={len(in_flight_jids)}")
    lines.append("")
    total_done = total_target = 0
    cells_done = 0
    cells_total = 0
    for dataset in DATASETS:
        for noise in NOISES:
            cells_total += 1
            cell = progress[(dataset, noise)]
            cell_done = sum(len(v["done"]) for v in cell["algos"].values())
            cell_target = len(ALGOS) * TASKS_PER_CELL_ALGO
            total_done += cell_done
            total_target += cell_target
            merge_tag = "merged" if cell["merged"] else "pending"
            if cell_done == cell_target and cell["merged"]:
                cells_done += 1
            lines.append(f"{dataset:<22} noise={noise}  partials={cell_done:>3}/{cell_target}  merge={merge_tag}")
            for algo in ALGOS:
                a = cell["algos"][algo]
                bar_len = 25
                filled = int(len(a["done"]) / TASKS_PER_CELL_ALGO * bar_len)
                bar = "█" * filled + "░" * (bar_len - filled)
                infl = f"  in-flight={a['in_flight']}" if a["in_flight"] else ""
                lines.append(f"    {algo:<5} {len(a['done']):>2}/{TASKS_PER_CELL_ALGO}  [{bar}]{infl}")
            lines.append("")
    lines.append(f"OVERALL: {total_done}/{total_target} partials  ({cells_done}/{cells_total} cells fully done)")
    REPORT_PATH.write_text("\n".join(lines))


def iteration(state: dict) -> bool:
    """Returns True if everything is complete (controller can exit)."""
    # 1. Refresh: get current in-flight array JIDs from squeue.
    in_flight = squeue_preorder_jids()

    # 2. Drop tracked arrays that have left the queue.
    for jid in list(state["arrays"].keys()):
        if jid not in in_flight:
            del state["arrays"][jid]

    # 3. Scan disk + decide submits per cell.
    free_slots = PREORDER_CAP - squeue_preorder_task_count()
    if free_slots < 0:
        free_slots = 0

    # Per-cell progress used for reporting + driving submits.
    progress: dict = {}
    for dataset, dconf in DATASETS.items():
        for noise in NOISES:
            cell = {"algos": {}, "merged": merged_on_disk(dconf["results_dir"], dataset, noise)}
            for algo in ALGOS:
                done = partials_on_disk(dconf["results_dir"], dataset, noise, algo)
                # Tasks currently in-flight for this cell-algo (by walking tracked arrays).
                inflight_rf: set[tuple[int, int]] = set()
                for jid, meta in state["arrays"].items():
                    if (meta["dataset"], meta["noise"], meta["algorithm"]) == (dataset, noise, algo):
                        for tid in meta["task_ids"]:
                            inflight_rf.add(rf_of(tid))
                cell["algos"][algo] = {
                    "done": done, "in_flight": len(inflight_rf), "inflight_rf": inflight_rf,
                }
            progress[(dataset, noise)] = cell

    write_report(progress, free_slots, in_flight)

    # 4. Submit pass — iterate cells in a deterministic order and consume free_slots.
    everything_done = True
    for dataset, dconf in DATASETS.items():
        for noise in NOISES:
            cell = progress[(dataset, noise)]
            for algo in ALGOS:
                done = cell["algos"][algo]["done"]
                inflight_rf = cell["algos"][algo]["inflight_rf"]
                missing = [
                    task_id_of(r, f)
                    for r in range(N_REPEATS) for f in range(N_FOLDS)
                    if (r, f) not in done and (r, f) not in inflight_rf
                ]
                if missing:
                    everything_done = False
                if not missing or free_slots == 0:
                    continue
                n = min(len(missing), free_slots)
                ids = missing[:n]
                jid = submit_split_array(
                    dataset, noise, algo, ids,
                    dconf["results_dir"], dconf["mem"], dconf["time"],
                )
                if jid:
                    state["arrays"][jid] = {
                        "dataset": dataset, "noise": noise, "algorithm": algo,
                        "task_ids": ids,
                    }
                    log(f"  submitted {dataset} n={noise} {algo} array={compact_ranges(ids)} → {jid}")
                    free_slots -= n

            # Merge: when all 100 partials exist for this cell and not merged yet → submit
            cell_done_count = sum(len(v["done"]) for v in cell["algos"].values())
            if cell_done_count == len(ALGOS) * TASKS_PER_CELL_ALGO and not cell["merged"]:
                # Avoid re-submitting if a merge job is already in flight for this cell.
                merge_key = f"merge_{dataset}_{noise}"
                if merge_key not in state.get("merges", {}):
                    if free_slots > 0:  # merge job is small but still counts
                        mjid = submit_merge(dataset, noise, dconf["results_dir"])
                        if mjid:
                            state.setdefault("merges", {})[merge_key] = mjid
                            log(f"  submitted MERGE {dataset} n={noise} → {mjid}")
                            free_slots -= 1
                            everything_done = False

    save_state(state)
    return everything_done


def main() -> int:
    log(f"=== master controller START (pid={os.getpid()}, poll={POLL_SECS}s) ===")
    log(f"matrix: {len(DATASETS)} datasets × {len(NOISES)} noise × {len(ALGOS)} algo × "
        f"{TASKS_PER_CELL_ALGO} (repeat,fold) = "
        f"{len(DATASETS) * len(NOISES) * len(ALGOS) * TASKS_PER_CELL_ALGO} target partials")
    state = load_state()
    while True:
        try:
            done = iteration(state)
        except Exception as e:  # noqa: BLE001
            log(f"  [iteration error] {e!r}")
            done = False
        if done:
            log("=== all cells fully done and merged; controller exiting ===")
            return 0
        log(f"poll done; sleeping {POLL_SECS}s")
        try:
            time.sleep(POLL_SECS)
        except KeyboardInterrupt:
            log("interrupted; exiting (in-flight SLURM jobs unaffected)")
            return 130


if __name__ == "__main__":
    sys.exit(main())
