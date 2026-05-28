"""Polling controller that fills training gaps for missing (dataset, learner)
cells outside the original master_controller scope.

Targets:
  - scene RF + LGBM
  - water_quality RF + LGBM
  - enron LGBM

Each cell = (dataset, base_learner, noise, algorithm). RF cells have
N_REPEATS=5, N_FOLDS=5 (25 tasks); LGBM cells have N_REPEATS=1, N_FOLDS=5
(5 tasks) because the orchestrator special-cases LightGBM to one repeat.

After all 4 algo partials of a (dataset, learner, noise) cell are on disk,
submits one merge+evaluate job for that cell.

Same QOS-cap throttling as master_controller (28 preorder tasks max).

Usage:
    nohup python scripts/ablations/fill_gaps_controller.py \\
        > slurm_logs/fill_gaps_controller.out 2>&1 &
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
STATE_FILE = Path("/tmp/preorder_fill_gaps_state.json")
LOG_PATH = ROOT / "slurm_logs" / "fill_gaps_controller.log"
REPORT_PATH = ROOT / "slurm_logs" / "fill_gaps_controller_report.txt"

POLL_SECS = int(os.environ.get("POLL_SECS", "600"))
PREORDER_CAP = 28
SLURM_USER = os.environ["USER"]

# Each cell entry: (dataset, learner) → resource profile + repeats/folds.
CELLS = {
    ("scene", "RF"): {
        "results_dir": "results/full_scene_split",
        "mem": "32G", "time": "08:00:00",
        "n_repeats": 5, "n_folds": 5,
    },
    ("scene", "LightGBM"): {
        "results_dir": "results/full_scene_lgbm_split",
        "mem": "32G", "time": "04:00:00",
        "n_repeats": 1, "n_folds": 5,
    },
    # water_quality already has all partials on disk (see merge script);
    # only needs local merge+eval, not SLURM training.
    ("enron", "LightGBM"): {
        "results_dir": "results/full_enron_lgbm_split",
        # Measured: BOPOS w/ joblib n_jobs=4 peaks at 33.5G (4× amplification
        # × 1378 pairwise LightGBM classifiers per worker). N_JOBS=1 drops
        # peak to ~8G but sequential. K=53 ILP without time limit takes
        # ~1h40min per IA × 8 IAs = 13h — way over the 8h slurm cap. Cap
        # each HiGHS solve at 5s (340 instances × 5s = ~28min per IA →
        # ~3.7h total). Near-optimal solutions are acceptable for the
        # paper-baseline comparison.
        # 32G needed: GLPK fallback (when HiGHS times out) uses more RAM
        # than HiGHS path; noise>=0.2 ILPs are larger and OOM'd at 16G.
        # n=0.3 ILPs are slowest (noisiest pairwise probas → more GLPK fallbacks);
        # 6h was hit before finishing IA1, bumped to 12h. Lower noises finish
        # well under 6h but inherit 12h cap for safety.
        "mem": "32G", "time": "12:00:00",
        "n_repeats": 1, "n_folds": 5,
        "n_jobs": 1,
        # n=0.3 IA1 alone exceeded 3h at HIGHS_TIME_LIMIT=5.0 (noisier probas
        # → more HiGHS timeouts → GLPK fallback dominates wall time). Drop to
        # 2.0s and accept 5% MIP relative gap so HiGHS itself returns a
        # near-optimal incumbent in time instead of falling back to GLPK.
        # n=0.0/0.1/0.2 currently running with the old setting; the new value
        # only applies to resubmits.
        # Tuned 2026-05-27: bump HiGHS budget to 10s + 10% gap so HiGHS
        # itself solves most instances (k=53 needs >2s to find feasibility),
        # and add a 60s GLPK cap so a single pathological instance can't
        # burn hours on the fallback path. n_test≈340; worst case
        # 340×60s = 5.7h per IA × 8 IAs only if EVERY instance falls back,
        # but typical fallback rate at these settings is ~10-20%.
        "highs_time_limit": 10.0,
        "highs_mip_rel_gap": 0.10,
        "glpk_time_limit": 60.0,
    },
}
NOISES = ["0.0", "0.1", "0.2", "0.3"]
ALGOS = ["bopos", "clr", "br", "cc"]


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
    return {"arrays": {}, "merges": {}}


def save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(STATE_FILE)


def _suffix_for(algo: str) -> str:
    return "" if algo == "bopos" else f"_{algo}"


def partials_on_disk(results_dir: str, dataset: str, noise: str, algo: str) -> set[tuple[int, int]]:
    suffix = _suffix_for(algo)
    pattern = str(
        ROOT / results_dir / f"dataset_{dataset}_noisy_{noise}{suffix}_r*_f*.pkl"
    )
    rx = re.compile(r".*_r(\d+)_f(\d+)\.pkl$")
    done: set[tuple[int, int]] = set()
    for p in glob.glob(pattern):
        m = rx.match(p)
        if m:
            done.add((int(m.group(1)), int(m.group(2))))
    return done


def merged_on_disk(results_dir: str, dataset: str, noise: str) -> bool:
    base = ROOT / results_dir
    for algo in ALGOS:
        sfx = _suffix_for(algo)
        if not (base / f"dataset_{dataset}_noisy_{noise}{sfx}.pkl").exists():
            return False
    return (base / f"evaluation_{dataset}_noisy_{noise}_dataset_level.csv").exists()


def squeue_preorder_jids() -> set[str]:
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


def submit_split_array(
    dataset: str, base_learner: str, noise: str, algorithm: str,
    task_ids: list[int], results_dir: str, mem: str, time_limit: str,
    n_jobs: int | None = None,
    highs_time_limit: float | None = None,
    highs_mip_rel_gap: float | None = None,
    glpk_time_limit: float | None = None,
) -> str | None:
    rng = compact_ranges(task_ids)
    env = (
        f"ALL,DATASET={dataset},NOISE_RATE={noise},RESULTS_DIR={results_dir},"
        f"ALGORITHM={algorithm},BASE_LEARNER={base_learner},SOLVER=highs"
    )
    if n_jobs is not None:
        env += f",N_JOBS={n_jobs}"
    if highs_time_limit is not None:
        env += f",HIGHS_TIME_LIMIT={highs_time_limit}"
    if highs_mip_rel_gap is not None:
        env += f",HIGHS_MIP_REL_GAP={highs_mip_rel_gap}"
    if glpk_time_limit is not None:
        env += f",GLPK_TIME_LIMIT={glpk_time_limit}"
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
        log(f"  [submit error] {dataset}/{base_learner} n={noise} {algorithm} array={rng}: "
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


def write_report(progress: dict, free_slots: int, in_flight: set[str]) -> None:
    lines = [
        f"# fill_gaps_controller report   {_dt.datetime.now().isoformat(timespec='seconds')}",
        f"# preorder cap={PREORDER_CAP}, free_slots={free_slots}, in_flight_arrays={len(in_flight)}",
        "",
    ]
    total_done = total_target = 0
    cells_done = cells_total = 0
    for (dataset, learner), conf in CELLS.items():
        n_tasks = conf["n_repeats"] * conf["n_folds"]
        for noise in NOISES:
            cells_total += 1
            cell = progress[(dataset, learner, noise)]
            cell_done = sum(len(v["done"]) for v in cell["algos"].values())
            cell_target = len(ALGOS) * n_tasks
            total_done += cell_done
            total_target += cell_target
            merge_tag = "merged" if cell["merged"] else "pending"
            if cell_done == cell_target and cell["merged"]:
                cells_done += 1
            lines.append(
                f"{dataset:<14}/{learner:<4} noise={noise}  "
                f"partials={cell_done:>3}/{cell_target}  merge={merge_tag}"
            )
            for algo in ALGOS:
                a = cell["algos"][algo]
                bar_len = 25
                filled = int(len(a["done"]) / n_tasks * bar_len) if n_tasks else 0
                bar = "#" * filled + "-" * (bar_len - filled)
                infl = f"  in-flight={a['in_flight']}" if a["in_flight"] else ""
                lines.append(f"    {algo:<5} {len(a['done']):>2}/{n_tasks}  [{bar}]{infl}")
            lines.append("")
    lines.append(f"OVERALL: {total_done}/{total_target} partials  ({cells_done}/{cells_total} cells fully done)")
    REPORT_PATH.write_text("\n".join(lines))


def iteration(state: dict) -> bool:
    in_flight = squeue_preorder_jids()

    for jid in list(state["arrays"].keys()):
        if jid not in in_flight:
            del state["arrays"][jid]

    free_slots = max(0, PREORDER_CAP - squeue_preorder_task_count())

    progress: dict = {}
    for (dataset, learner), conf in CELLS.items():
        n_repeats = conf["n_repeats"]
        n_folds = conf["n_folds"]
        for noise in NOISES:
            cell = {"algos": {}, "merged": merged_on_disk(conf["results_dir"], dataset, noise)}
            for algo in ALGOS:
                done = partials_on_disk(conf["results_dir"], dataset, noise, algo)
                inflight_rf: set[tuple[int, int]] = set()
                for jid, meta in state["arrays"].items():
                    if (meta["dataset"], meta["learner"], meta["noise"], meta["algorithm"]) == (
                        dataset, learner, noise, algo
                    ):
                        for tid in meta["task_ids"]:
                            r, f = divmod(tid, n_folds)
                            inflight_rf.add((r, f))
                cell["algos"][algo] = {
                    "done": done, "in_flight": len(inflight_rf), "inflight_rf": inflight_rf,
                    "n_repeats": n_repeats, "n_folds": n_folds,
                }
            progress[(dataset, learner, noise)] = cell

    write_report(progress, free_slots, in_flight)

    everything_done = True
    for (dataset, learner), conf in CELLS.items():
        n_repeats = conf["n_repeats"]
        n_folds = conf["n_folds"]
        n_tasks = n_repeats * n_folds
        for noise in NOISES:
            cell = progress[(dataset, learner, noise)]
            for algo in ALGOS:
                done = cell["algos"][algo]["done"]
                inflight_rf = cell["algos"][algo]["inflight_rf"]
                missing = [
                    r * n_folds + f
                    for r in range(n_repeats) for f in range(n_folds)
                    if (r, f) not in done and (r, f) not in inflight_rf
                ]
                if missing:
                    everything_done = False
                if not missing or free_slots == 0:
                    continue
                n = min(len(missing), free_slots)
                ids = missing[:n]
                jid = submit_split_array(
                    dataset, learner, noise, algo, ids,
                    conf["results_dir"], conf["mem"], conf["time"],
                    n_jobs=conf.get("n_jobs"),
                    highs_time_limit=conf.get("highs_time_limit"),
                    highs_mip_rel_gap=conf.get("highs_mip_rel_gap"),
                    glpk_time_limit=conf.get("glpk_time_limit"),
                )
                if jid:
                    state["arrays"][jid] = {
                        "dataset": dataset, "learner": learner, "noise": noise,
                        "algorithm": algo, "task_ids": ids,
                    }
                    log(f"  submitted {dataset}/{learner} n={noise} {algo} array={compact_ranges(ids)} -> {jid}")
                    free_slots -= n

            cell_done_count = sum(len(v["done"]) for v in cell["algos"].values())
            if not cell["merged"]:
                everything_done = False
            if cell_done_count == len(ALGOS) * n_tasks and not cell["merged"]:
                merge_key = f"merge_{dataset}_{learner}_{noise}"
                if merge_key not in state.get("merges", {}):
                    if free_slots > 0:
                        mjid = submit_merge(dataset, noise, conf["results_dir"])
                        if mjid:
                            state.setdefault("merges", {})[merge_key] = mjid
                            log(f"  submitted MERGE {dataset}/{learner} n={noise} -> {mjid}")
                            free_slots -= 1

    save_state(state)
    return everything_done


def main() -> int:
    log(f"=== fill_gaps_controller START (pid={os.getpid()}, poll={POLL_SECS}s) ===")
    target_total = sum(
        len(NOISES) * len(ALGOS) * conf["n_repeats"] * conf["n_folds"]
        for conf in CELLS.values()
    )
    log(f"matrix: {len(CELLS)} (dataset,learner) cells x {len(NOISES)} noise x "
        f"{len(ALGOS)} algo -> {target_total} target partials")
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
