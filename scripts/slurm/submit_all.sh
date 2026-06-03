#!/usr/bin/env bash
# Submit the full RF + LightGBM regeneration matrix to SLURM.
#
# Cluster cap (QOSMaxSubmitJobPerUser) counts each ARRAY TASK separately, so a
# 25-task array consumes 25 slots. The controller pre-computes the full sbatch
# list, then drips submissions: it waits until expanded squeue count is at
# least `array_size` below SUBMIT_CAP before each train submit. sbatch failures
# with "QOSMaxSubmitJobPerUserLimit" trigger an exponential backoff retry.
#
# Enron throttle: enron's 25-task ~17h BOPOS arrays otherwise eat the whole cap.
# Live enron tasks are held to ENRON_QUEUE_FRACTION (default 0.4) of SUBMIT_CAP —
# enron arrays are split into per-repeat 5-task sub-arrays to fit — so other
# datasets keep ~60% of the queue. The throttle lifts automatically once every
# non-enron cell (both learners) is complete on disk. Run the RF and LightGBM
# controllers CONCURRENTLY (LEARNERS=...) so the protected 60% is actually used.
#
# Eval jobs are submitted with --dependency=afterok:<comma-list-of-train-jobids>.

set -uo pipefail   # NB: not -e — sbatch may fail transiently and we want to retry

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
    shift
fi

STAMP="$(date +%Y%m%d)"
RESULTS_DIR_RF="${1:-results/rerun_${STAMP}_rf}"
RESULTS_DIR_LGBM="${2:-results/rerun_${STAMP}_lgbm}"

mkdir -p slurm_logs "${RESULTS_DIR_RF}" "${RESULTS_DIR_LGBM}"

# DATASETS env var (space-separated) overrides default list — useful for
# adding a single dataset (e.g. viruspseaac) without interrupting a running
# controller. Both controllers share the squeue cap via wait_for_room().
if [ -n "${DATASETS:-}" ]; then
    read -ra DATASETS <<< "${DATASETS}"
else
    DATASETS=(chd_49 emotions scene yeast water_quality gpositivepseaac plantpseaac viruspseaac humanpseaac enron)
fi

# results_manager.py writes pickles as `dataset_<lowercase(config.name)>_noisy_…`
# where config.name is the display name (e.g. "Water-quality"). For most keys the
# lowercased display name equals the key, but water_quality differs (hyphen).
declare -A DS_FILENAME=(
    [water_quality]=water-quality
)
ds_filename() {
    local k="$1"
    echo "${DS_FILENAME[$k]:-$k}"
}
NOISE_RATES=(0.0 0.1 0.2 0.3)
ALGOS=(bopos clr br cc ecc)

SUBMIT_CAP=40         # def QoS MaxSubmitPU
SAFETY_MARGIN=2       # keep this many slots free for eval submits

# Enron throttle. Enron (K=53) BOPOS arrays are 25 long-running (~17h) tasks that
# otherwise monopolise the shared QoS cap and starve the rest of the matrix. We
# hold the number of *enron* tasks live in the queue to a fraction of SUBMIT_CAP
# so the other datasets always keep the remaining ~60%. To honour a 40% cap with
# 25-task arrays, enron arrays are split into per-repeat 5-task sub-arrays (see
# submit_array). The throttle auto-lifts once every NON-enron cell (both learners)
# is complete on disk — at which point enron may use the full queue.
#   Intended use: run an RF (enron-only, now) controller and an LGBM controller
#   CONCURRENTLY; both consult the same live squeue count, so the cap holds
#   across controllers. ENRON_QUEUE_FRACTION / ENRON_BUDGET are env-overridable.
ENRON_QUEUE_FRACTION="${ENRON_QUEUE_FRACTION:-0.4}"
ENRON_BUDGET="${ENRON_BUDGET:-$(awk "BEGIN{printf \"%d\", ${SUBMIT_CAP} * ${ENRON_QUEUE_FRACTION}}")}"

# Canonical full dataset list — used by all_others_done() so the throttle-lift
# check is robust to a DATASETS env override that narrows this controller's work.
ALL_DATASETS=(chd_49 emotions scene yeast water_quality gpositivepseaac plantpseaac viruspseaac humanpseaac enron)

count_jobs() {
    # squeue -r expands arrays — same as SLURM's accounting unit.
    squeue -u "$USER" -h -r 2>/dev/null | wc -l
}

count_enron_jobs() {
    # Live enron tasks across ALL controllers/learners (job names are
    # tr_enron_n<rate>_<algo>_<learner>). Drives the enron budget gate.
    squeue -u "$USER" -h -r -o "%j" 2>/dev/null | grep -c '_enron_' || true
}

# True once every non-enron cell is complete on disk for BOTH learners. When this
# holds there is nothing left for enron to starve, so the throttle lifts.
all_others_done() {
    local ds noise algo learner rdir
    for ds in "${ALL_DATASETS[@]}"; do
        [ "${ds}" = "enron" ] && continue
        for learner in RF LightGBM; do
            if [ "${learner}" = "RF" ]; then rdir="${RESULTS_DIR_RF}"; else rdir="${RESULTS_DIR_LGBM}"; fi
            for noise in "${NOISE_RATES[@]}"; do
                for algo in "${ALGOS[@]}"; do
                    cell_done "${ds}" "${noise}" "${algo}" "${learner}" "${rdir}" || return 1
                done
            done
        done
    done
    return 0
}

# Block until adding `need` enron tasks keeps total live enron <= ENRON_BUDGET.
# Lifts immediately once all non-enron work is complete. Re-checks the (cheap-ish
# but not free) all_others_done() at most every ~5 min while waiting.
wait_for_enron_room() {
    local need="$1" i=0
    while :; do
        if [ $(( i % 10 )) -eq 0 ] && all_others_done; then
            return 0
        fi
        local e
        e=$(count_enron_jobs)
        if [ $(( e + need )) -le "${ENRON_BUDGET}" ]; then
            return 0
        fi
        i=$(( i + 1 ))
        sleep 30
    done
}

wait_for_room() {
    local need="$1"
    while :; do
        local n
        n=$(count_jobs)
        if [ $(( n + need + SAFETY_MARGIN )) -le "${SUBMIT_CAP}" ]; then
            return 0
        fi
        sleep 30
    done
}

# Glob suffix for an algo's per-fold pickles, mirroring how the writers name
# files. clr/br/cc/bopos follow results_manager.py with no learner token. ECC is
# inconsistent: RF ecc files are dataset_<ds>_noisy_<r>_ecc_r*_f*.pkl (no token)
# while LGBM ecc files carry a learner token (..._ecc_lgbm_r*_f*.pkl). A single
# "_ecc*" glob matches both; the callers expand the pattern UNQUOTED. Without
# this, completed ECC cells are never detected and get re-run every pass.
algo_suffix() {
    local algo="$1" learner="$2"
    case "${algo}" in
        clr) echo "_clr"  ;;
        br)  echo "_br"   ;;
        cc)  echo "_cc"   ;;
        ecc) echo "_ecc*" ;;
        *)   echo ""       ;;
    esac
}

# Returns 0 if all expected split pickles for this cell already exist on disk.
cell_done() {
    local dataset="$1" noise="$2" algo="$3" learner="$4" results_dir="$5"
    local expected_n
    if [ "${learner}" = "LightGBM" ]; then expected_n=5; else expected_n=25; fi
    local suffix
    suffix=$(algo_suffix "${algo}" "${learner}")
    local file_ds
    file_ds=$(ds_filename "${dataset}")
    local pattern="${results_dir}/dataset_${file_ds}_noisy_${noise}${suffix}_r*_f*.pkl"
    local n
    n=$(ls ${pattern} 2>/dev/null | wc -l)
    [ "${n}" -ge "${expected_n}" ]
}

# Returns 0 if all 5 folds of one repeat of a cell already exist. Used to skip
# already-complete per-repeat chunks when enron arrays are split.
chunk_done() {
    local dataset="$1" noise="$2" algo="$3" learner="$4" results_dir="$5" repeat="$6"
    local suffix
    suffix=$(algo_suffix "${algo}" "${learner}")
    local file_ds
    file_ds=$(ds_filename "${dataset}")
    # NB: pattern expanded UNQUOTED so the "_ecc*" glob (and r*/f*) match.
    local pattern="${results_dir}/dataset_${file_ds}_noisy_${noise}${suffix}_r${repeat}_f*.pkl"
    local n
    n=$(ls ${pattern} 2>/dev/null | wc -l)
    [ "${n}" -ge 5 ]
}

# Submit one array, retrying on QOSMaxSubmitJobPerUserLimit. Echoes job ID on stdout.
submit_array() {
    local dataset="$1" noise="$2" algo="$3" learner="$4" results_dir="$5"
    if cell_done "${dataset}" "${noise}" "${algo}" "${learner}" "${results_dir}"; then
        echo "[skip ${dataset} n${noise} ${algo} ${learner}] all splits present" >&2
        echo ""    # empty jid → no dep added
        return 0
    fi
    local array_size array_spec
    if [ "${learner}" = "LightGBM" ]; then
        array_spec="0-4"; array_size=5
    else
        array_spec="0-24"; array_size=25
    fi
    local jobname="tr_${dataset}_n${noise}_${algo}_${learner}"
    # High-K datasets need more RAM for BOPOS: it stores all K*(K-1)/2 pairwise
    # proba matrices for inference. Enron (K=53) -> 1378 pairs. With the ILP
    # search now running 16-way parallel (see train.sbatch NCORES), each worker
    # forks a copy of that data and peak RSS hit ~67G -> 64G OOM-killed. Compute
    # nodes have ~1.5T RAM, so request 160G (2.4x headroom) and keep the fast
    # 16-way search (~2h/task) rather than throttling parallelism.
    local extra=()
    if [ "${dataset}" = "enron" ] && [ "${algo}" = "bopos" ]; then
        # 360G. enron's K=53 ILP has ~140k transitivity constraints, so the GLPK
        # LP-relaxation basis factorization plateaus at ~220 GiB at 8-way (job
        # 158550); HiGHS instead grew UNBOUNDED past any cap, which is why
        # enron-BOPOS uses GLPK + a per-instance GLPK_TIME_LIMIT (train.sbatch).
        # The old 240G cap was below GLPK's plateau and OOM-killed the noisy folds.
        # QOS ceiling: mem=384G AND cpu=64, with MaxMemPerCPU=6000 — so 384G
        # implies ~66 CPUs and trips QOSMaxCpuPerJobLimit; 360G (~62 CPUs) is the
        # safe max. The search runs 12-way (ENRON_SEARCH_N_JOBS) — peaks under
        # 360G while ~50% faster than the 8-way default.
        extra+=(--mem=360G --time=48:00:00)
    fi
    # Chunk plan: list of "spec:size" array sub-specs. Normally one full array.
    # Enron arrays larger than ENRON_BUDGET are split into per-repeat 5-task
    # sub-arrays (train.sbatch derives repeat=TASK_ID/5, fold=TASK_ID%5, so an
    # --array=R*5..R*5+4 sub-array runs repeat R, folds 0-4). This is what lets
    # the 40% throttle hold a 25-task enron array to <=40% of the queue.
    local -a chunks=()
    if [ "${dataset}" = "enron" ] && [ "${array_size}" -gt "${ENRON_BUDGET}" ]; then
        local reps=$(( array_size / 5 )) r
        for (( r=0; r<reps; r++ )); do
            chunks+=("$(( r * 5 ))-$(( r * 5 + 4 )):5")
        done
    else
        chunks+=("${array_spec}:${array_size}")
    fi

    local all_jids="" chunk
    for chunk in "${chunks[@]}"; do
        local spec="${chunk%%:*}" size="${chunk##*:}"
        # Skip already-complete per-repeat chunks (only meaningful when split).
        if [ "${#chunks[@]}" -gt 1 ]; then
            local rep=$(( ${spec%%-*} / 5 ))
            if chunk_done "${dataset}" "${noise}" "${algo}" "${learner}" "${results_dir}" "${rep}"; then
                echo "[skip ${jobname} r${rep}] splits present" >&2
                continue
            fi
        fi
        local cmd=(sbatch
            --job-name="${jobname}"
            --array="${spec}"
            "${extra[@]}"
            --export=ALL,DATASET="${dataset}",NOISE_RATE="${noise}",RESULTS_DIR="${results_dir}",ALGORITHM="${algo}",BASE_LEARNER="${learner}"
            scripts/slurm/train.sbatch
        )

        if [ "${DRY_RUN}" -eq 1 ]; then
            printf 'DRY: %s\n' "${cmd[*]}" >&2
            all_jids="${all_jids:+${all_jids} }DRY_${dataset}_${noise}_${algo}_${learner}_a${spec}"
            continue
        fi

        local attempt=0 backoff=30 jid=""
        while :; do
            # Enron-specific gate first (keeps enron <=ENRON_BUDGET until all
            # other work is done), then the global room gate.
            [ "${dataset}" = "enron" ] && wait_for_enron_room "${size}"
            wait_for_room "${size}"
            local out
            if out=$("${cmd[@]}" 2>&1); then
                jid=$(echo "${out}" | awk '/Submitted batch job/ {print $NF}')
                if [ -n "${jid}" ]; then
                    echo "[${jobname} a${spec}] -> ${jid}" >&2
                    break
                fi
            fi
            attempt=$(( attempt + 1 ))
            echo "[${jobname} a${spec}] sbatch attempt ${attempt} failed: ${out}" >&2
            if [ "${attempt}" -ge 20 ]; then
                echo "[${jobname} a${spec}] giving up after 20 attempts" >&2
                jid=""
                break
            fi
            sleep "${backoff}"
            [ "${backoff}" -lt 300 ] && backoff=$(( backoff * 2 ))
        done
        [ -n "${jid}" ] && all_jids="${all_jids:+${all_jids} }${jid}"
    done

    # Echo all chunk job ids (space-separated) for eval dependency collection.
    echo "${all_jids}"
    return 0
}

submit_eval() {
    local dataset="$1" results_dir="$2" deps="$3"
    local jobname="ev_${dataset}_$(basename "${results_dir}")"

    # Filter deps to only jobs still pending/running. SLURM rejects
    # --dependency=afterok on jobs already in COMPLETED state (purged
    # from active controller state), with "Job dependency problem".
    local active_deps=""
    local jid
    for jid in ${deps//,/ }; do
        if squeue -h -j "${jid}" 2>/dev/null | grep -q .; then
            active_deps="${active_deps:+${active_deps},}${jid}"
        fi
    done

    local cmd=(sbatch
        --job-name="${jobname}"
    )
    if [ -n "${active_deps}" ]; then
        cmd+=(--dependency=afterany:"${active_deps}")
    fi
    cmd+=(--export=ALL,DATASET="${dataset}",RESULTS_DIR="${results_dir}"
        scripts/slurm/eval.sbatch
    )

    if [ "${DRY_RUN}" -eq 1 ]; then
        printf 'DRY: %s\n' "${cmd[*]}" >&2
        return 0
    fi

    local attempt=0 backoff=30
    while :; do
        wait_for_room 1
        local out
        if out=$("${cmd[@]}" 2>&1); then
            local jid
            jid=$(echo "${out}" | awk '/Submitted batch job/ {print $NF}')
            if [ -n "${jid}" ]; then
                echo "[${jobname}] -> ${jid}" >&2
                return 0
            fi
        fi
        attempt=$(( attempt + 1 ))
        echo "[${jobname}] sbatch attempt ${attempt} failed: ${out}" >&2
        if [ "${attempt}" -ge 20 ]; then
            echo "[${jobname}] giving up after 20 attempts" >&2
            return 1
        fi
        sleep "${backoff}"
        [ "${backoff}" -lt 300 ] && backoff=$(( backoff * 2 ))
    done
}

run_learner() {
    local learner="$1" results_dir="$2"
    echo "==========================================" >&2
    echo "BASE_LEARNER=${learner}  RESULTS_DIR=${results_dir}" >&2
    echo "==========================================" >&2

    local dataset noise algo
    for dataset in "${DATASETS[@]}"; do
        # SKIP_CELLS env var: space-separated "<learner>:<dataset>" tokens to skip
        # entirely (training + eval), for resuming after a partial run.
        if [ -n "${SKIP_CELLS:-}" ] && [[ " ${SKIP_CELLS} " == *" ${learner}:${dataset} "* ]]; then
            echo "[skip] ${learner}:${dataset} (already done)" >&2
            continue
        fi
        local dataset_jobs=()
        for noise in "${NOISE_RATES[@]}"; do
            for algo in "${ALGOS[@]}"; do
                local jid
                jid=$(submit_array "${dataset}" "${noise}" "${algo}" "${learner}" "${results_dir}") || continue
                # submit_array may echo several chunk job ids (enron split) —
                # add each as its own dependency (unquoted split on whitespace).
                [ -n "${jid}" ] && dataset_jobs+=(${jid})
            done
        done
        if [ "${#dataset_jobs[@]}" -gt 0 ]; then
            local deps
            deps=$(IFS=,; echo "${dataset_jobs[*]}")
            submit_eval "${dataset}" "${results_dir}" "${deps}"
        fi
    done
}

echo "Controller starting at $(date -Is)" >&2
# LEARNERS env var (space-separated subset of "RF LightGBM") restricts which
# learners this controller processes — useful for launching a parallel LGBM-only
# controller while the main controller is still finishing RF.
LEARNERS="${LEARNERS:-RF LightGBM}"
for learner in ${LEARNERS}; do
    case "${learner}" in
        RF)       run_learner RF       "${RESULTS_DIR_RF}"   ;;
        LightGBM) run_learner LightGBM "${RESULTS_DIR_LGBM}" ;;
        *) echo "[warn] unknown learner: ${learner}" >&2 ;;
    esac
done

echo "" >&2
echo "=== ALL SUBMITTED at $(date -Is) ===" >&2
echo "Watch progress:  squeue -u \$USER" >&2
echo "Logs:            slurm_logs/" >&2
echo "RF results:      ${RESULTS_DIR_RF}" >&2
echo "LGBM results:    ${RESULTS_DIR_LGBM}" >&2
echo "" >&2
echo "After all evals finish, aggregate:" >&2
echo "  python -m preorder4mlc.utils.summarize_metrics --results_dir ${RESULTS_DIR_RF}   --output_dir ${RESULTS_DIR_RF}_summary" >&2
echo "  python -m preorder4mlc.utils.summarize_metrics --results_dir ${RESULTS_DIR_LGBM} --output_dir ${RESULTS_DIR_LGBM}_summary" >&2
