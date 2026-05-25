#!/usr/bin/env bash
# Sync local paper_revision/ into the Overleaf clone under experiment_revision/,
# then commit and push to Overleaf.
#
# Usage:
#   scripts/sync_overleaf.sh                  # rsync + commit + push
#   scripts/sync_overleaf.sh --dry-run        # show what would change
#   scripts/sync_overleaf.sh --no-push        # rsync + commit only
#   scripts/sync_overleaf.sh -m "msg"         # custom commit message

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/paper_revision/"
DEST_REPO="$ROOT/overleaf"
DEST="$DEST_REPO/experiment_revision/"

DRY_RUN=0
DO_PUSH=1
MSG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --no-push) DO_PUSH=0; shift ;;
    -m) MSG="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ ! -d "$SRC" ]]; then
  echo "source missing: $SRC" >&2; exit 1
fi
if [[ ! -d "$DEST_REPO/.git" ]]; then
  echo "overleaf clone missing: $DEST_REPO" >&2; exit 1
fi

mkdir -p "$DEST"

RSYNC_FLAGS=(-av --delete)
[[ $DRY_RUN -eq 1 ]] && RSYNC_FLAGS+=(--dry-run)

rsync "${RSYNC_FLAGS[@]}" "$SRC" "$DEST"

[[ $DRY_RUN -eq 1 ]] && { echo "(dry run; no commit)"; exit 0; }

cd "$DEST_REPO"
git add experiment_revision

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
