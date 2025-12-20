# model_api/app.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np

app = FastAPI()

MODEL_PATH = "model.pt"
MAXLEN = 200

class Req(BaseModel):
    method: str
    path: str
    headers: dict
    body: str
    query: str

class ConvClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        e = self.embed(x).permute(0, 2, 1)
        c = torch.relu(self.conv(e))
        p = self.pool(c).squeeze(-1)
        return torch.sigmoid(self.fc(p))

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
vocab = checkpoint["vocab"]
model = ConvClassifier(vocab_size=len(vocab) + 1)
model.load_state_dict(checkpoint["model"])
model.eval()

def encode(text):
    arr = [vocab.get(c, 0) for c in text[:MAXLEN]]
    arr += [0] * (MAXLEN - len(arr))
    return torch.tensor([arr], dtype=torch.long)

@app.post("/score")
def score(req: Req):
    text = f"{req.method} {req.path} {req.query} {req.body}".lower()
    x = encode(text)
    with torch.no_grad():
        s = model(x).item()
    return {"score": round(s, 3), "info": "ml-model"}
