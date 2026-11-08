import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

df = pd.read_csv("data/raw.csv", keep_default_na=False)
df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip().str.lower()
df = df[df["label"].isin(["negative", "neutral", "positive"])]
df = df[df["text"].str.len() > 0]

# Dedupe on normalized text so near-identical rows can't leak across splits
df["key"] = df["text"].str.lower().str.replace(r"\s+", " ", regex=True)
df = df.drop_duplicates("key").drop(columns="key").reset_index(drop=True)

train, tmp = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
val, test = train_test_split(tmp, test_size=0.5, stratify=tmp["label"], random_state=SEED)

train.to_csv("data/train.csv", index=False)
val.to_csv("data/val.csv", index=False)
test.to_csv("data/test.csv", index=False)
print(len(train), len(val), len(test))
print(test["label"].value_counts(normalize=True))
