#!/bin/bash
# Fail unless the filesystem holding $1 keeps >= MIN_FREE_FRAC free (and >= MIN_FREE_GB, if set) after writing $2 GB.
# Usage: scripts/disk_guard.sh <path> <needed_GB>     (MIN_FREE_FRAC default 0.30, override via env; MIN_FREE_GB optional)
# On shared volumes that are already fuller than the fraction allows, set MIN_FREE_FRAC
# to what is achievable and MIN_FREE_GB to an absolute floor; both are enforced.
set -eu
path=${1:-.}; need_gb=${2:-0}; frac=${MIN_FREE_FRAC:-0.30}; min_gb=${MIN_FREE_GB:-0}
mkdir -p "$path"
read size avail <<<"$(df --output=size,avail -B1 "$path" | tail -1)"
floor=$(python3 -c "print(max(int($size*$frac), int($min_gb*1e9)))")
after=$(python3 -c "print(int($avail - $need_gb*1e9))")
if [ "$after" -lt "$floor" ]; then
  echo "disk_guard: refusing to write ${need_gb} GB to $path: free after = $((after/1000000000)) GB < floor $((floor/1000000000)) GB (${frac} of $((size/1000000000)) GB)" >&2
  exit 3
fi
echo "disk_guard: ok ($((avail/1000000000)) GB free, floor $((floor/1000000000)) GB, need ${need_gb} GB)"
