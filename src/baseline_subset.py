import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from common import L2I

tr = pd.read_csv("data/train.csv", keep_default_na=False)
te = pd.read_csv("data/test.csv", keep_default_na=False).head(300)
pipe = make_pipeline(
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=2000, class_weight="balanced"),
)
pipe.fit(tr.text, tr.label.map(L2I))
pred = pipe.predict(te.text)
print("baseline macro F1 on first 300 test rows:",
      round(f1_score(te.label.map(L2I), pred, average="macro"), 4))
