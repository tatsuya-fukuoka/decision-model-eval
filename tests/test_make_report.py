"""report.md の生成（src.make_report）：テンプレートの差し込み位置がすべて解決され、表の書式が正しいこと。"""

import re

from src import make_report


def test_md_table_shape():
    t = make_report.md(["a", "b"], [[1, 2], [3, 4]])
    lines = t.splitlines()
    assert lines[0] == "| a | b |" and lines[1] == "| --- | ---: |" and len(lines) == 4


def test_ci_format_and_undefined():
    assert make_report.ci({"x": 0.5, "x_ci_low": 0.4, "x_ci_high": 0.6}, "x") == "0.500（0.400–0.600）"
    assert make_report.ci({"x": float("nan")}, "x") == "未定義"
    assert make_report.ci({"x": 0.5}, "x") == "0.500"


def test_build_resolves_all_placeholders():
    text = make_report.build()
    assert not re.search(r"\{\{table:", text)
    assert "自前の業務データでの評価は行っていません" in text
