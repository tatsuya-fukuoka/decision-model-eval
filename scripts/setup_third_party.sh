#!/usr/bin/env bash
# 外部リポジトリを固定コミットで third_party/ に取得する（git 管理外）。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p third_party
clone() {  # name url commit
  local dir="third_party/$1"
  if [ ! -d "$dir/.git" ]; then git clone -q "$2" "$dir"; fi
  git -C "$dir" fetch -q origin "$3" 2>/dev/null || true
  git -C "$dir" checkout -q "$3"
  echo "$1 @ $(git -C "$dir" rev-parse HEAD)"
}
clone S1MB https://github.com/hotchpotch/S1MB.git dbfc74699b814cadc8e3384ffa8705a39958cc91
clone sokudan https://github.com/hiroki-abe-58/sokudan.git eccded37532d8a9afe5d6715ab62231a24366e65
