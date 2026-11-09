"""ソースを AST で静的に走査する（実行しない）。危険になりうる import・呼び出し・文字列を列挙する。

    python3 -I scripts/static_scan.py <dir>

1行目に JSON の要約（ファイル数・行数・トップレベルの import）、2行目以降に該当箇所（file:line: 内容）。
"""
import ast
import collections
import json
import os
import re
import sys

root = sys.argv[1]
RISKY_MODULES = {"socket", "urllib", "requests", "httpx", "http", "aiohttp", "ftplib", "smtplib", "paramiko",
                 "websocket", "websockets", "subprocess", "pty", "ctypes", "cffi", "pickle", "marshal", "shutil",
                 "tempfile", "importlib", "huggingface_hub", "multiprocessing", "asyncio", "ssl", "base64", "zlib",
                 "codecs", "signal", "threading", "concurrent", "platform", "getpass", "pathlib", "glob", "io"}
RISKY_CALLS = {"eval", "exec", "compile", "__import__", "open", "getattr", "setattr", "globals", "locals", "vars",
               "breakpoint", "input"}
RISKY_ATTRS = {"system", "popen", "execv", "execve", "execl", "execvp", "spawnv", "spawnl", "fork", "remove",
               "unlink", "rmdir", "rmtree", "environ", "getenv", "putenv", "load", "loads", "load_library",
               "urlopen", "connect", "Popen", "run", "check_output", "call", "chmod", "chown", "kill",
               "inline_asm_elementwise", "extern_elementwise"}
URL = re.compile(r"(https?|ftp|wss?)://|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")

imports = collections.Counter()
hits = []
n_files = n_lines = 0
for dirpath, _, files in os.walk(root):
    for f in sorted(files):
        if not f.endswith(".py"):
            continue
        path = os.path.join(dirpath, f)
        rel = os.path.relpath(path, root)
        src = open(path, encoding="utf-8").read()
        n_files += 1
        n_lines += src.count("\n")
        tree = ast.parse(src, filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    top = a.name.split(".")[0]
                    imports[a.name] += 1
                    if top in RISKY_MODULES:
                        hits.append((rel, node.lineno, f"import {a.name}"))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if node.level == 0:
                    imports[mod] += 1
                    if mod.split(".")[0] in RISKY_MODULES:
                        hits.append((rel, node.lineno, f"from {mod} import {[a.name for a in node.names]}"))
            elif isinstance(node, ast.Call):
                fn = node.func
                if isinstance(fn, ast.Name) and fn.id in RISKY_CALLS:
                    hits.append((rel, node.lineno, f"call {fn.id}(...): {ast.unparse(node)[:160]}"))
                elif isinstance(fn, ast.Attribute) and fn.attr in RISKY_ATTRS:
                    hits.append((rel, node.lineno, f"call .{fn.attr}: {ast.unparse(node)[:160]}"))
            elif isinstance(node, ast.Attribute) and node.attr in {"environ", "modules", "__builtins__", "__dict__"}:
                hits.append((rel, node.lineno, f"attr .{node.attr}: {ast.unparse(node)[:160]}"))
            elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
                v = node.value if isinstance(node.value, str) else node.value.decode("latin1")
                if URL.search(v):
                    hits.append((rel, node.lineno, f"string with URL/IP: {v[:160]!r}"))
                if len(v) > 400 and not v.lstrip().startswith(("\n", " ")) and " " not in v[:200]:
                    hits.append((rel, node.lineno, f"long opaque string ({len(v)} chars)"))

print(json.dumps({"files": n_files, "lines": n_lines,
                  "top_level_imports": sorted({k.split('.')[0] for k in imports})}, ensure_ascii=False))
for h in hits:
    print(f"{h[0]}:{h[1]}: {h[2]}")
