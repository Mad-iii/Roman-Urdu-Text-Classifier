import time, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from common import summarize, L2I

tr, te = pd.read_csv("data/train.csv", keep_default_na=False), pd.read_csv("data/test.csv", keep_default_na=False)
pipe = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000, class_weight="balanced"),
)
pipe.fit(tr.text, tr.label.map(L2I))

lat = []
for t in te.text[:300]:
    s = time.perf_counter(); pipe.predict([t]); lat.append((time.perf_counter() - s) * 1000)

pred = pipe.predict(te.text)
HOURLY = 0.05  # assumed CPU $/hour, state this in README
cost = (sum(lat) / len(lat) / 1000) * 1000 / 3600 * HOURLY
summarize("tfidf_lr", te.label.map(L2I), pred, lat, cost)
