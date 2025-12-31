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

# SU DUNG KIEN TRUC LSTM (Layer 2) THEO BAI BAO
# Paper section 4.5: Bidirectional LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # Su dung LSTM 2 chieu (bidirectional=True)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Output cua Bi-LSTM la hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (Batch, Sequence Length)
        e = self.embed(x)
        # LSTM tra ve (output, (hidden, cell))
        lstm_out, _ = self.lstm(e)
        # Lay gia tri trung binh cua tat ca cac time steps (Mean Pooling)
        out = torch.mean(lstm_out, dim=1)
        return self.sigmoid(self.fc(out))

# Load model (Luu y: Can train lai model bang file train_simple.py truoc khi chay)
checkpoint = torch.load(MODEL_PATH, map_location="cpu")
vocab = checkpoint["vocab"]
model = LSTMClassifier(vocab_size=len(vocab) + 1)
model.load_state_dict(checkpoint["model"])
model.eval()

def encode(text):
    arr = [vocab.get(c, 0) for c in text[:MAXLEN]]
    arr += [0] * (MAXLEN - len(arr))
    return torch.tensor([arr], dtype=torch.long)

@app.post("/score")
def score(req: Req):
    # Ket hop method, path, query, body thanh 1 chuoi de dua vao LSTM
    text = f"{req.method} {req.path} {req.query} {req.body}".lower()
    x = encode(text)
    with torch.no_grad():
        s = model(x).item()
    return {"score": round(s, 3), "info": "lstm-model"}