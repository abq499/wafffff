# train_simple.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import re
import csv
from pathlib import Path

# --- Dataset loader (very simple, expects CSV with 'text','label') ---
class TextDataset(Dataset):
    def __init__(self, texts, labels, maxlen=200, vocab=None):
        self.texts = texts
        self.labels = labels
        self.maxlen = maxlen
        # simple char-level vocab
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

# --- simple model: embedding -> 1D conv -> pool -> fc ---
class ConvClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.conv = nn.Conv1d(embed_dim, 128, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # x: (B, L)
        e = self.embed(x).permute(0,2,1)  # (B, embed, L)
        c = torch.relu(self.conv(e))
        p = self.pool(c).squeeze(-1)
        out = torch.sigmoid(self.fc(p))
        return out

# --- toy training loop ---
if __name__ == "__main__":
    import csv
    from pathlib import Path

    data_path = Path("../data/labeled_requests.csv")

    texts = []
    labels = []

    with data_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = f"{row['method']} {row['path']} {row['query']} {row['body']}"
            label = 1 if row["label"].strip().lower() == "attack" else 0
            texts.append(text)
            labels.append(label)

    print("Total samples:", len(texts))
    print("Attack samples:", sum(labels))
    print("Normal samples:", len(labels) - sum(labels))

    ds = TextDataset(texts, labels, maxlen=200)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    model = ConvClassifier(vocab_size=len(ds.vocab) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for epoch in range(10):
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
        {
            "model": model.state_dict(),
            "vocab": ds.vocab
        },
        "model.pt"
    )

    print("✅ Model trained & saved as model.pt")

