import json
from dataclasses import asdict

import pytest

from src.data import bench_ja, s1mb


def s1mb_row():
    return {
        "case_id": "c1", "group_id": "GROUP-SECRET", "split": "test",
        "provenance_json": '{"source":"PROV-SECRET"}', "legacy_aux_json": "{}",
        "input": {
            "state_json": json.dumps({"premise": "p", "hypothesis": "h"}, ensure_ascii=False),
            "decisions": [
                {"id": "judgment", "type": "choice", "instructions_json": '"選んでください"', "system_prompt": "",
                 "criteria": [{"id": "entailment", "description_json": '"含意"', "value": None},
                              {"id": "contradiction", "description_json": '"矛盾"', "value": None},
                              {"id": "neutral", "description_json": '{"text":"中立"}', "value": None}]},
                {"id": "other", "type": "noul", "instructions_json": '"x"', "system_prompt": "",
                 "criteria": [{"id": "false", "description_json": '"no"', "value": None},
                              {"id": "true", "description_json": '"yes"', "value": None}]},
            ],
        },
        "targets": [
            {"decision_id": "judgment", "ids": ["neutral", "entailment", "contradiction"], "probabilities": [0.25, 0.75, 0.0]},
            {"decision_id": "other", "ids": ["false", "true"], "probabilities": [1.0, 0.0]},
        ],
    }


def test_parse_row_keeps_order_and_hides_labels():
    req, targets = s1mb.parse_row(s1mb_row(), "jnli-choice", "choice")
    assert len(req.questions) == 1  # 記述子の task と同じ型だけ
    q = req.questions[0]
    assert [o.id for o in q.options] == ["entailment", "contradiction", "neutral"]
    assert q.instructions == "選んでください"
    assert q.options[2].description == '{"text": "中立"}'
    assert req.state == {"premise": "p", "hypothesis": "h"}
    dumped = json.dumps(asdict(req), ensure_ascii=False)
    for secret in ("GROUP-SECRET", "PROV-SECRET", "0.75"):
        assert secret not in dumped
    (t,) = targets
    assert t.option_ids == ("entailment", "contradiction", "neutral")
    assert t.probs == (0.75, 0.0, 0.25)  # criteria 順に揃え、ソフトターゲットは丸めない
    assert t.group_id == "GROUP-SECRET"


def test_parse_row_rejects_mismatched_targets():
    row = s1mb_row()
    row["targets"][0]["ids"] = ["a", "b", "c"]
    with pytest.raises(ValueError):
        s1mb.parse_row(row, "jnli-choice", "choice")


def test_bench_ja_tasks(tmp_path):
    path = tmp_path / "b.jsonl"
    items = [
        {"item_id": "i0", "state": "本文0", "department": "技術", "urgency": 2, "churn": True, "industry": "SECRET-ATTR"},
        {"item_id": "i1", "state": "本文1", "department": "その他", "urgency": 0, "churn": False, "industry": "x"},
    ]
    path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in items), encoding="utf-8")
    reqs, tg = bench_ja.load_task("bench_ja-department", path)
    assert [o.id for o in reqs[0].questions[0].options] == ["請求", "技術", "営業", "その他"]
    assert tg[0].probs == (0.0, 1.0, 0.0, 0.0) and tg[1].probs == (0.0, 0.0, 0.0, 1.0)
    assert "SECRET-ATTR" not in json.dumps(asdict(reqs[0]), ensure_ascii=False)
    reqs, tg = bench_ja.load_task("bench_ja-urgency", path)
    assert tg[0].values == (0.0, 1.0, 2.0) and tg[0].probs == (0.0, 0.0, 1.0)
    reqs, tg = bench_ja.load_task("bench_ja-churn", path)
    assert tg[0].option_ids == ("false", "true") and tg[0].probs == (0.0, 1.0)


@pytest.mark.skipif(not (s1mb.DEFAULT_ROOT / "categories").is_dir(), reason="S1MB データ未取得")
def test_real_s1mb_counts():
    ids = s1mb.load_category()
    assert len(ids) == 27
    total = sum(len(s1mb.load_benchmark(t)[2]) for t in ids)
    assert total == 8067
    assert sum(t.startswith("generalization-") for t in ids) == 8


@pytest.mark.skipif(not bench_ja.DEFAULT_PATH.is_file(), reason="bench_ja 未取得")
def test_real_bench_ja_counts():
    for task in bench_ja.TASKS:
        reqs, tg = bench_ja.load_task(task)
        assert len(reqs) == len(tg) == 300
