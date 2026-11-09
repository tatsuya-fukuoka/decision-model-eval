#!/usr/bin/env bash
# 自作アダプタが使う PyPI パッケージを固定版・ハッシュ確認付きで data/packages/ に取得する（実行はしない）。
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/packages
fetch() {  # url sha256 filename
  curl -sSfL -o "data/packages/$3" "$1"
  echo "$2  data/packages/$3" | sha256sum -c -
}
fetch https://files.pythonhosted.org/packages/58/73/c9527f45530d77411f6a2fa0a576909299f4738373966c880b99e73474af/strands_decider-0.3.0-py3-none-any.whl \
  b5817ffb469127f611ca375f9c6342ec232aac27267405dcba19d0c2b7a100b1 strands_decider-0.3.0-py3-none-any.whl
