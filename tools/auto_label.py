# tools/auto_label.py
import csv
from pathlib import Path

in_path = Path("data/requests.csv")
out_path = Path("data/labeled_requests.csv")

with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", newline="", encoding="utf-8") as fout:
    reader = csv.DictReader(fin)
    fieldnames = reader.fieldnames + ["label"]
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    for row in reader:
        action = (row.get("action") or "").lower()
        if action.startswith("blocked"):
            label = "attack"
        else:
            label = "normal"
        row["label"] = label
        writer.writerow(row)

print("Wrote labeled file:", out_path.resolve())
