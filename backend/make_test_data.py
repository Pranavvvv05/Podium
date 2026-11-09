import csv
import os
import zipfile

import numpy as np
from PIL import Image
from sklearn.datasets import fetch_20newsgroups, load_digits

OUT = "test_data"
os.makedirs(OUT, exist_ok=True)

digits = load_digits()
per_class = 150
counts = {d: 0 for d in range(10)}
zip_path = os.path.join(OUT, "digits_images.zip")
with zipfile.ZipFile(zip_path, "w") as z:
    for img, label in zip(digits.images, digits.target):
        if counts[label] >= per_class:
            continue
        pic = Image.fromarray((img / 16 * 255).astype(np.uint8)).resize((32, 32))
        path = os.path.join(OUT, "tmp.png")
        pic.save(path)
        z.write(path, f"digits/digit_{label}/{counts[label]}.png")
        counts[label] += 1
os.remove(os.path.join(OUT, "tmp.png"))
print("created", zip_path, sum(counts.values()), "images")

cats = ["rec.autos", "sci.space", "rec.sport.hockey", "comp.graphics"]
news = fetch_20newsgroups(subset="train", categories=cats, remove=("headers", "footers", "quotes"))
rows = []
for text, label in zip(news.data, news.target):
    text = " ".join(text.split())
    if len(text) >= 40:
        rows.append((text[:1500], news.target_names[label]))
rows = rows[:2000]
csv_path = os.path.join(OUT, "news_text.csv")
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["text", "category"])
    w.writerows(rows)
print("created", csv_path, len(rows), "rows")