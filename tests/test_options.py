from src.data.options import anonymize, argmax_id, permutations_for, permute, restore
from src.data.types import Option, Question


def q(n=4, qtype="choice"):
    return Question(id="q", type=qtype, instructions="x",
                    options=tuple(Option(id=f"id{i}", description=f"d{i}") for i in range(n)))


def test_anonymize_keeps_meaning_and_order():
    question = q()
    anon, mapping = anonymize(question)
    assert [o.description for o in anon.options] == [o.description for o in question.options]
    assert [mapping[o.id] for o in anon.options] == [o.id for o in question.options]
    probs = {o.id: p for o, p in zip(anon.options, [0.1, 0.2, 0.3, 0.4])}
    assert restore(probs, mapping) == {"id0": 0.1, "id1": 0.2, "id2": 0.3, "id3": 0.4}


def test_noul_not_anonymized():
    noul = Question(id="n", type="noul", instructions="x",
                    options=(Option("false", "no"), Option("true", "yes")))
    anon, mapping = anonymize(noul)
    assert anon == noul and mapping == {"false": "false", "true": "true"}


def test_restore_rejects_missing_keys():
    import pytest
    with pytest.raises(ValueError):
        restore({"option_0": 1.0}, {"option_0": "a", "option_1": "b"})


def test_permutation_roundtrip():
    question = q(5)
    perms = permutations_for(question, 5, seed=7)
    assert len(perms) == 5 and len(set(perms)) == 5
    assert tuple(range(5)) not in perms
    assert perms == permutations_for(question, 5, seed=7)  # 決定的
    for order in perms:
        permuted = permute(question, order)
        # 並べ替え -> 匿名化 -> モデル出力（位置ごと） -> 復元 で、元IDの確率が正しく戻る
        anon, mapping = anonymize(permuted)
        position_probs = [0.05, 0.1, 0.15, 0.3, 0.4]
        returned = {o.id: p for o, p in zip(anon.options, position_probs)}
        restored = restore(returned, mapping)
        for pos, idx in enumerate(order):
            assert restored[f"id{idx}"] == position_probs[pos]
        # 並べ替えても、元の選択肢順で同点解決する
        assert argmax_id(restored, [o.id for o in question.options]) == f"id{order[-1]}"


def test_permutations_limited_by_factorial():
    assert len(permutations_for(q(2), 5, seed=0)) == 1
    assert len(permutations_for(q(3), 5, seed=0)) == 5
