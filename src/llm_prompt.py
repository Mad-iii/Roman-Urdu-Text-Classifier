import time, random, pandas as pd
from concurrent.futures import ThreadPoolExecutor
import anthropic
from common import summarize, L2I, LABELS

MODEL = "claude-haiku-5-5"
PRICE_IN, PRICE_OUT = 0.0, 0.0  # USD per 1M tokens: fill from the current pricing page

tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
random.seed(0)
shots = pd.concat([tr[tr.label == l].sample(3, random_state=0) for l in LABELS]).sample(frac=1, random_state=0)
examples = "\n".join(f"Message: {r.text}\nLabel: {r.label}" for r in shots.itertuples())

SYSTEM = (
    "You classify Roman Urdu (Urdu written in Latin script) messages by sentiment. "
    "Reply with exactly one word: negative, neutral, or positive.\n\nExamples:\n" + examples
)
client = anthropic.Anthropic()

def classify(text):
    s = time.perf_counter()
    r = client.messages.create(model=MODEL, max_tokens=5, system=SYSTEM,
                               messages=[{"role": "user", "content": f"Message: {text}\nLabel:"}])
    ms = (time.perf_counter() - s) * 1000
    out = r.content[0].text.strip().lower()
    lab = next((l for l in LABELS if l in out), "neutral")  # unparseable -> neutral; count these
    bad = not any(l in out for l in LABELS)
    return L2I[lab], ms, r.usage.input_tokens, r.usage.output_tokens, bad

with ThreadPoolExecutor(4) as ex:
    rows = list(ex.map(classify, te.text))

pred, lat, tin, tout, bad = map(list, zip(*rows))
cost_1k = (sum(tin) * PRICE_IN + sum(tout) * PRICE_OUT) / 1e6 / len(te) * 1000
summarize("llm_fewshot", te.label.map(L2I), pred, lat, cost_1k, {"unparseable": sum(bad)})