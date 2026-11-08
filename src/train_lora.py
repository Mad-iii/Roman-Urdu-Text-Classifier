import pandas as pd, mlflow
from datasets import Dataset
from sklearn.metrics import f1_score
from peft import LoraConfig, get_peft_model
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, DataCollatorWithPadding)
from common import L2I, I2L

BASE = "Qwen/Qwen2.5-0.5B"  # or 1.5B if your GPU has room
tok = AutoTokenizer.from_pretrained(BASE)
if tok.pad_token is None: tok.pad_token = tok.eos_token

def load(n):
    d = pd.read_csv(f"data/{n}.csv", keep_default_na=False); d["labels"] = d.label.map(L2I)
    return Dataset.from_pandas(d[["text", "labels"]]).map(
        lambda b: tok(b["text"], truncation=True, max_length=128), batched=True)

train, val = load("train"), load("val")
model = AutoModelForSequenceClassification.from_pretrained(
    BASE, num_labels=3, id2label=I2L, label2id=L2I)  # fp32 weights; fp16 autocast below
model.config.pad_token_id = tok.pad_token_id
model = get_peft_model(model, LoraConfig(
    task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir="runs/lora", num_train_epochs=3, learning_rate=2e-4,
    per_device_train_batch_size=16, per_device_eval_batch_size=32, warmup_ratio=0.05,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    metric_for_best_model="macro_f1", fp16=True, report_to="mlflow", seed=42,
    gradient_accumulation_steps=2, save_total_limit=1,
)
mlflow.set_experiment("roman-urdu")
Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=lambda p: {"macro_f1": f1_score(p.label_ids, p.predictions.argmax(-1), average="macro")}
        ).train()
model.save_pretrained("models/lora"); tok.save_pretrained("models/lora")
