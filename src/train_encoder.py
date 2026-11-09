import numpy as np, pandas as pd, mlflow
from datasets import Dataset
from sklearn.metrics import f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from common import L2I, LABELS, I2L

BASE = "xlm-roberta-base"
tok = AutoTokenizer.from_pretrained(BASE)

def load(n):
    d = pd.read_csv(f"data/{n}.csv", keep_default_na=False); d["labels"] = d.label.map(L2I)
    return Dataset.from_pandas(d[["text", "labels"]]).map(
        lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

train, val = load("train"), load("val")  # test is NOT loaded here
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=3, id2label=I2L, label2id=L2I)

def metrics(p):
    return {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}

args = TrainingArguments(
    output_dir="runs/encoder", num_train_epochs=4, learning_rate=2e-5,
    per_device_train_batch_size=32, per_device_eval_batch_size=64, warmup_steps=0.1,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="macro_f1", fp16=True, report_to="mlflow", seed=42,
    save_total_limit=1,
)
mlflow.set_experiment("roman-urdu")
Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
        data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics).train()
model.save_pretrained("models/encoder"); tok.save_pretrained("models/encoder")
