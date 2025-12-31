# notebooks/train_simple.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
from pathlib import Path

# --- Dataset loader ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, maxlen=200, vocab=None):
        self.texts = texts
        self.labels = labels
        self.maxlen = maxlen
        if vocab is None:
            chars = sorted(list({c for t in texts for c in t}))
            self.vocab = {c:i+1 for i,c in enumerate(chars)}
            self.vocab['<pad>'] = 0
        else:
            self.vocab = vocab

    def encode(self, s):
        arr = [self.vocab.get(c,0) for c in s[:self.maxlen]]
        if len(arr) < self.maxlen:
            arr += [0]*(self.maxlen-len(arr))
        return np.array(arr, dtype=np.int64)

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        return torch.tensor(self.encode(self.texts[idx])), torch.tensor(self.labels[idx], dtype=torch.float32)

# --- KIEN TRUC LSTM GIONG TRONG MODEL_API ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        e = self.embed(x)
        lstm_out, _ = self.lstm(e)
        out = torch.mean(lstm_out, dim=1)
        return self.sigmoid(self.fc(out))

if __name__ == "__main__":
    # Duong dan data
    data_path = Path("../data/labeled_requests.csv")
    if not data_path.exists():
        data_path = Path("data/labeled_requests.csv")

    texts = []
    labels = []

    try:
        with data_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = f"{row['method']} {row['path']} {row['query']} {row['body']}"
                label = 1 if row["label"].strip().lower() == "attack" else 0
                texts.append(text)
                labels.append(label)
    except Exception:
        print("Loi: Khong tim thay file data. Hay chay tools/jsonl_to_csv.py truoc.")
        exit()

    print(f"So luong mau: {len(texts)}")

    ds = TextDataset(texts, labels, maxlen=200)
    dl = DataLoader(ds, batch_size=16, shuffle=True)

    model = LSTMClassifier(vocab_size=len(ds.vocab) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    print("Bat dau train LSTM...")
    for epoch in range(15): # Tang epoch len 15 vi LSTM hoi tu cham hon
        total_loss = 0
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | loss={total_loss/len(dl):.4f}")

    torch.save(
        {"model": model.state_dict(), "vocab": ds.vocab},
        "model.pt"
    )
    print("Da luu model.pt thanh cong (Kien truc LSTM)")