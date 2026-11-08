import json, os
import numpy as np
from sklearn.metrics import f1_score, classification_report

LABELS = ["negative", "neutral", "positive"]
L2I = {l: i for i, l in enumerate(LABELS)}
I2L = {i: l for l, i in L2I.items()}


def summarize(name, y_true, y_pred, latencies_ms, cost_per_1k, extra=None):
    res = {
        "approach": name,
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro")), 4),
        "latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 1),
        "latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 1),
        "cost_per_1k_usd": round(float(cost_per_1k), 5),
    }
    if extra:
        res.update(extra)
    print(json.dumps(res, indent=2))
    print(classification_report(y_true, y_pred, target_names=LABELS, digits=3))
    os.makedirs("results", exist_ok=True)
    with open(f"results/{name}.json", "w") as f:
        json.dump(res, f, indent=2)
    return res