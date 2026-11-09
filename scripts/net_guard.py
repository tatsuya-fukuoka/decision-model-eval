"""Python の audit hook でネットワーク接続を拒否してから、別のスクリプトを実行する（カーネル一致確認用）。

    python -I scripts/net_guard.py --log GUARD.jsonl SCRIPT.py [args...]

- 拒否：AF_UNIX 以外のソケットの connect / sendto / sendmsg / bind、名前解決（getaddrinfo など）。
  拒否した呼び出しは PermissionError になり、--log に記録する。
- 記録のみ（拒否しない）：プロセスの起動（subprocess.Popen、os.system、os.exec*、os.posix_spawn）。
  Triton は GPU 用のコンパイルで ptxas や C コンパイラを起動するので、止めると実行できない。
- 対象は Python から行う操作だけ。ネイティブコード（C 拡張）が直接行う通信は、audit hook では捕まえられない。
- このスクリプトは本リポジトリの src を import しない。
"""

from __future__ import annotations

import json
import os
import runpy
import socket
import sys
import time
import traceback

NET_EVENTS = {"socket.connect", "socket.sendto", "socket.sendmsg", "socket.bind"}
DNS_EVENTS = {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo"}
PROC_EVENTS = {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork"}


def main() -> None:
    if len(sys.argv) < 4 or sys.argv[1] != "--log":
        raise SystemExit("usage: net_guard.py --log GUARD.jsonl SCRIPT.py [args...]")
    log_path, script = sys.argv[2], sys.argv[3]
    log = open(log_path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115

    def write(kind: str, event: str, detail: str) -> None:
        rec = {"t": time.time(), "pid": os.getpid(), "kind": kind, "event": event, "detail": detail[:300]}
        if kind == "blocked":  # 呼び出し元を特定できるように、直近のスタック（ファイル:行 関数）を残す
            rec["stack"] = [f"{f.filename}:{f.lineno} {f.name}" for f in traceback.extract_stack(limit=12)[:-2]]
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def hook(event: str, args: tuple) -> None:
        if event in NET_EVENTS:
            sock = args[0] if args else None
            family = getattr(sock, "family", None)
            if family == socket.AF_UNIX:
                write("allowed_unix", event, repr(args[1:])[:200])
                return
            write("blocked", event, repr(args[1:])[:200])
            raise PermissionError(f"network blocked by net_guard: {event}")
        if event in DNS_EVENTS:
            write("blocked", event, repr(args)[:200])
            raise PermissionError(f"name resolution blocked by net_guard: {event}")
        if event in PROC_EVENTS:
            exe = args[0] if args else ""
            write("process", event, repr(exe)[:300])

    sys.addaudithook(hook)
    write("start", "net_guard", script)
    sys.argv = [script, *sys.argv[4:]]
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()
