#!/usr/bin/env bash
# 公式 uv.lock が PyPI から取る NVIDIA の wheel を、同じ sha256 のファイルを持つ PyTorch のミラーから取得する。
# PyPI が遅い環境向けの任意の手順。lock のハッシュと一致したファイルだけを data/wheels/ に残す。
# 使い方: bash scripts/fetch_cuda_wheels.sh envs/<name>/requirements.lock.txt
set -euo pipefail
cd "$(dirname "$0")/.."
req="$1"
mkdir -p data/wheels
for pkg in nvidia-nvshmem-cu13 nvidia-cusparse nvidia-cufft nvidia-curand nvidia-cublas nvidia-cusolver \
           nvidia-cudnn-cu13 nvidia-nccl-cu13 nvidia-cusparselt-cu13 nvidia-cuda-nvrtc; do
  hashes=$(awk -v p="$pkg==" 'index($0,p)==1{f=1;next} f&&/--hash=sha256:/{sub(/.*sha256:/,"");sub(/[ \\].*/,"");print;next} f{f=0}' "$req")
  if [ -z "$hashes" ]; then continue; fi
  page=$(curl -sSf --max-time 60 "https://download.pytorch.org/whl/$pkg/")
  ok=""
  for h in $hashes; do
    href=$(echo "$page" | grep -o "href=\"[^\"]*#sha256=$h\"" | grep "linux[^\"]*x86_64" | head -1 | sed 's/href="//;s/#sha256=.*//') || true
    if [ -z "$href" ]; then continue; fi
    case "$href" in
      /*) url="https://download.pytorch.org$href" ;;
      http*) url="$href" ;;
      *) url="https://download.pytorch.org/whl/$pkg/$href" ;;
    esac
    f="data/wheels/$(basename "$url" | sed 's/%2B/+/g')"
    [ -f "$f" ] || curl -sSfL --max-time 1800 -o "$f" "$url"
    if echo "$h  $f" | sha256sum -c - >/dev/null; then echo "ok $f"; ok=1; break; fi
    rm -f "$f"
  done
  [ -n "$ok" ] || echo "no mirror match: $pkg (stays on PyPI)"
done
