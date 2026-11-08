import pandas as pd

def norm(s): return s.str.lower().str.replace(r"\s+", " ", regex=True)

def test_no_leakage():
    tr, va, te = (pd.read_csv(f"data/{n}.csv", keep_default_na=False) for n in ("train", "val", "test"))
    t, v, x = set(norm(tr.text)), set(norm(va.text)), set(norm(te.text))
    assert not (t & x) and not (v & x) and not (t & v)

def test_label_balance_similar():
    tr, te = pd.read_csv("data/train.csv", keep_default_na=False), pd.read_csv("data/test.csv", keep_default_na=False)
    diff = (tr.label.value_counts(normalize=True) - te.label.value_counts(normalize=True)).abs().max()
    assert diff < 0.02
