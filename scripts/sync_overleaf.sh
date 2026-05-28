#!/usr/bin/env bash
# Sync local paper_revision/ into the Overleaf clone under experiment_revision/,
# then commit and push to Overleaf.
#
# Usage:
#   scripts/sync_overleaf.sh                  # rsync + commit + push
#   scripts/sync_overleaf.sh --dry-run        # show what would change
#   scripts/sync_overleaf.sh --no-push        # rsync + commit only
#   scripts/sync_overleaf.sh -m "msg"         # custom commit message
#   scripts/sync_overleaf.sh --watch          # auto-sync on changes (requires fswatch)
#   scripts/sync_overleaf.sh --watch --interval 10   # debounce window in seconds (default 5)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/paper_revision/"
DEST_REPO="$ROOT/overleaf"
DEST="$DEST_REPO/experiment_results/"

DRY_RUN=0
DO_PUSH=1
WATCH=0
INTERVAL=5
MSG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --no-push) DO_PUSH=0; shift ;;
    --watch) WATCH=1; shift ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    -m) MSG="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ $WATCH -eq 1 ]]; then
  if ! command -v fswatch >/dev/null 2>&1; then
    echo "fswatch not found. install with: brew install fswatch" >&2
    exit 1
  fi
  if [[ ! -d "$SRC" ]]; then
    echo "source missing: $SRC" >&2; exit 1
  fi
  echo "watching $SRC (debounce ${INTERVAL}s). ctrl-c to stop."
  SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
  CHILD_ARGS=()
  [[ $DO_PUSH -eq 0 ]] && CHILD_ARGS+=(--no-push)
  # initial sync
  "$SELF" "${CHILD_ARGS[@]}" || echo "initial sync failed; continuing to watch"
  # fswatch -o emits one line per debounced batch
  fswatch -o --latency "$INTERVAL" --event Created --event Updated --event Removed --event Renamed "$SRC" \
    | while read -r _; do
        echo "[$(date +%H:%M:%S)] change detected, syncing..."
        "$SELF" "${CHILD_ARGS[@]}" || echo "sync failed; continuing to watch"
      done
  exit 0
fi

if [[ ! -d "$SRC" ]]; then
  echo "source missing: $SRC" >&2; exit 1
fi
if [[ ! -d "$DEST_REPO/.git" ]]; then
  echo "overleaf clone missing: $DEST_REPO" >&2; exit 1
fi

mkdir -p "$DEST"

RSYNC_FLAGS=(
  -av --delete --delete-excluded
  --exclude='*.aux'
  --exclude='*.log'
  --exclude='*.out'
  --exclude='*.fdb_latexmk'
  --exclude='*.fls'
  --exclude='*.synctex.gz'
  --exclude='*.toc'
  # Exclude all figures_original/ — intermediate raw output, not needed on Overleaf.
  # Only figs/ (final figures) is synced.
  --exclude='figures_original/'
  --exclude='figures_enhanced/'
  # LGBM panel PDFs are large; we only keep what §3 abstain overview
  # actually references (the 2 summary_grid figures under abstain/_shared/).
  # Everything else under lgbm/ is excluded to stay under Overleaf's
  # 2000-file project limit. Local paper_revision/ still has all files.
  --exclude='figures_original/lgbm/aggregate/'
  --exclude='figures_enhanced/lgbm/aggregate/'
  --exclude='figures_original/lgbm/abstain/f1_pa/'
  --exclude='figures_enhanced/lgbm/abstain/f1_pa/'
  --exclude='figures_original/lgbm/abstain/jaccard_pa/'
  --exclude='figures_enhanced/lgbm/abstain/jaccard_pa/'
  --exclude='figures_original/lgbm/abstain/_shared/per_dataset/'
  --exclude='figures_enhanced/lgbm/abstain/_shared/per_dataset/'
)
[[ $DRY_RUN -eq 1 ]] && RSYNC_FLAGS+=(--dry-run)

rsync "${RSYNC_FLAGS[@]}" "$SRC" "$DEST"

[[ $DRY_RUN -eq 1 ]] && { echo "(dry run; no commit)"; exit 0; }

cd "$DEST_REPO"
git add experiment_results

if git diff --cached --quiet; then
  echo "no changes to commit"
  exit 0
fi

if [[ -z "$MSG" ]]; then
  MSG="update experiment_revision ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
fi

git commit -m "$MSG"

if [[ $DO_PUSH -eq 1 ]]; then
  git push origin master
fi
