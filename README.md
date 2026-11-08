# Roman Urdu Sentiment Classifier

| Approach | Macro F1 | p50 latency | Cost / 1k msgs |
|---|---|---|---|
| TF-IDF + LR (baseline) | | | |
| Few-shot LLM (claude-haiku-5-5) | | | |
| XLM-R base, fine-tuned | | | |
| Qwen2.5-0.5B + LoRA | | | |

Test set: N messages, held out before any code was written. CPU cost assumes $0.05/hr.

## Run it
`docker compose up` then the curl example.  Demo: <link/GIF>

## What still fails
(paste 8-10 rows from results/*_errors.csv and group them: sarcasm, code-switching, negation, spelling variants)

## Data & license
...
## Honest notes
(what surprised you, limits, variance across seeds)