import os, torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.getenv("MODEL_DIR", "/models/best")  # local path or HF Hub id
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))

app = FastAPI(title="Roman Urdu classifier")

class Req(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=64)

@app.get("/health")
def health(): return {"status": "ok", "model": MODEL_DIR}

@app.post("/predict")
def predict(req: Req):
    enc = tok(req.texts, return_tensors="pt", truncation=True, max_length=128, padding=True)
    with torch.no_grad():
        probs = model(**enc).logits.softmax(-1)
    out = []
    for p in probs:
        i = int(p.argmax())
        out.append({"label": model.config.id2label[i], "confidence": round(float(p[i]), 4)})
    return {"predictions": out}