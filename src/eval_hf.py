import sys, time, torch, pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from common import summarize, L2I, LABELS

# usage: python src/eval_hf.py encoder models/encoder
#        python src/eval_hf.py lora models/lora Qwen/Qwen2.5-0.5B
name, path = sys.argv[1], sys.argv[2]
base = sys.argv[3] if len(sys.argv) > 3 else None
HOURLY = 0.05  # assumed CPU $/hour, same as baseline

tok = AutoTokenizer.from_pretrained(path)
if base:
    from peft import PeftModel
    model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=3)
    model.config.pad_token_id = tok.pad_token_id
    model = PeftModel.from_pretrained(model, path).merge_and_unload()
else:
    model = AutoModelForSequenceClassification.from_pretrained(path)
model.eval().to("cpu")

te = pd.read_csv("data/test.csv")
preds, lat = [], []
with torch.no_grad():
    for t in te.text:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=128)
        s = time.perf_counter()
        logits = model(**enc).logits
        lat.append((time.perf_counter() - s) * 1000)
        preds.append(int(logits.argmax(-1)))

cost = (sum(lat) / len(lat) / 1000) * 1000 / 3600 * HOURLY
summarize(name, te.label.map(L2I).tolist(), preds, lat, cost)

# "What still fails": dump errors for the README
err = te.assign(pred=[LABELS[p] for p in preds])
err[err.label != err.pred].to_csv(f"results/{name}_errors.csv", index=False)