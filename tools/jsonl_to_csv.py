import json
import csv
from pathlib import Path

in_path = Path(__file__).parent.parent / "data" / "requests.jsonl"
out_path = Path(__file__).parent.parent / "data" / "requests.csv"

if not in_path.exists():
    print(f"❌ Không tìm thấy file: {in_path}")
    exit()

rows = []
with in_path.open("r", encoding="utf-8") as fin:
    for line in fin:
        try:
            obj = json.loads(line)
            rows.append(obj)
        except Exception as e:
            print("⚠️ Lỗi đọc dòng:", e)

if not rows:
    print("❌ Không có dữ liệu trong log.")
    exit()

# Lấy tất cả key
keys = sorted({k for row in rows for k in row.keys()})

with out_path.open("w", newline="", encoding="utf-8") as fout:
    writer = csv.DictWriter(fout, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)

print(f"✅ Đã ghi {len(rows)} dòng vào {out_path}")
