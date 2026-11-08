import os, time, json
import pandas as pd
from openai import OpenAI
from common import summarize, L2I, LABELS

BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
KEY = os.environ["LLM_API_KEY"]
RPM = int(os.getenv("RPM", "10"))   # requests per minute, stay under the limit
N = int(os.getenv("N", "0"))        # 0 = whole test set

os.makedirs("results", exist_ok=True)
CACHE = "results/llm_cache_" + MODEL.replace("/", "_").replace(":", "_") + ".jsonl"

tr = pd.read_csv("data/train.csv", keep_default_na=False)
te = pd.read_csv("data/test.csv", keep_default_na=False)
if N:
    te = te.head(N)  # test.csv is already shuffled, so head() is a random sample

shots = pd.concat([tr[tr.label == l].sample(3, random_state=0) for l in LABELS]).sample(frac=1, random_state=0)
examples = "\n".join(f"Message: {r.text}\nLabel: {r.label}" for r in shots.itertuples())
SYSTEM = (
    "You classify Roman Urdu (Urdu written in Latin script) messages by sentiment. "
    "Reply with exactly one word: negative, neutral, or positive.\n\nExamples:\n" + examples
)

client = OpenAI(base_url=BASE_URL, api_key=KEY)

cache = {}
if os.path.exists(CACHE):
    with open(CACHE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cache[r["i"]] = r

def call(text):
    kwargs = {}
    if "gpt-oss" in MODEL:
        kwargs["extra_body"] = {"reasoning_effort": "low"}
    for attempt in range(5):
        try:
            s = time.perf_counter()
            r = client.chat.completions.create(
                model=MODEL, max_tokens=200, temperature=0,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"Message: {text}\nLabel:"}],
                **kwargs)
            ms = (time.perf_counter() - s) * 1000
            return (r.choices[0].message.content or ""), ms
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                print("rate limited, waiting...", attempt + 1)
                time.sleep(20 * (attempt + 1))
                continue
            raise
    return None, None

with open(CACHE, "a", encoding="utf-8") as f:
    for i, row in enumerate(te.itertuples()):
        if i in cache:
            continue
        t0 = time.perf_counter()
        out, ms = call(row.text)
        if out is None:
            print("Still rate limited (probably the daily cap). Stopping; rerun later to resume.")
            break
        rec = {"i": i, "out": out, "ms": ms}
        cache[i] = rec
        f.write(json.dumps(rec) + "\n"); f.flush()
        if len(cache) % 25 == 0:
            print(f"{len(cache)}/{len(te)} done")
        time.sleep(max(0, 60 / RPM - (time.perf_counter() - t0)))

idx = sorted(i for i in cache if i < len(te))
print(f"Evaluated {len(idx)} of {len(te)} test rows")
if not idx:
    raise SystemExit("No results yet.")

y_true, pred, lat, bad = [], [], [], 0
for i in idx:
    out = cache[i]["out"].strip().lower()
    hit = next((l for l in LABELS if l in out), None)
    bad += hit is None
    pred.append(L2I[hit or "neutral"])  # unparseable counted as neutral, reported below
    y_true.append(L2I[te.iloc[i].label])
    lat.append(cache[i]["ms"])

name = "llm_" + MODEL.split("/")[-1].replace(":", "_")
summarize(name, y_true, pred, lat, 0.0,
          {"n_evaluated": len(idx), "unparseable": int(bad), "free_tier": True})