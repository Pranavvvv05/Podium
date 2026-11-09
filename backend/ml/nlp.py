import pickle
import time
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

try:
    from ml.deep import TorchTextClassifier
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MAX_ROWS = 20000
MIN_PER_CLASS = 10


def write_text_csv(src_csv: str, text_col: str, label_col: str, out_csv: str) -> dict:
    """Make a clean 2-column (text, label) file. Returns {label: count}."""
    df = pd.read_csv(src_csv)
    for col in (text_col, label_col):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Columns: {list(df.columns)}")
    if text_col == label_col:
        raise ValueError("Text column and label column must be different")
    df = df[[text_col, label_col]].dropna()
    df.columns = ["text", "label"]
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str)
    df = df[df["text"] != ""]
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42)
    counts = df["label"].value_counts()
    if len(counts) < 2:
        raise ValueError("Need at least 2 different labels")
    if len(counts) > 50:
        raise ValueError(f"Label column has {len(counts)} different values: it does not look like a class column")
    small = [str(c) for c, n in counts.items() if n < MIN_PER_CLASS]
    if small:
        raise ValueError(f"Each label needs at least {MIN_PER_CLASS} rows. Too few for: {', '.join(small)}")
    df.to_csv(out_csv, index=False)
    return {str(k): int(v) for k, v in counts.items()}


def prepare_text_data(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    X, y = df["text"].astype(str).to_numpy(), df["label"].astype(str).to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    return {"problem_type": "text_classification", "n_rows": len(X), "n_features": 1,
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def get_text_models() -> dict:
    models = {
        "ANN": (
            Pipeline([("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)),
                      ("model", MLPClassifier(max_iter=200, early_stopping=True, random_state=42))]),
            {"model__hidden_layer_sizes": [(128,), (128, 64)], "model__alpha": [1e-4, 1e-2]},
        ),
    }
    if HAS_TORCH:
        grid = {"model__lr": [0.005]}
        models["RNN"] = (Pipeline([("model", TorchTextClassifier(arch="gru"))]), grid)
        models["TextCNN"] = (Pipeline([("model", TorchTextClassifier(arch="textcnn"))]), grid)
    return models


def train_one_text(name: str, data: dict, n_jobs: int = 1) -> dict:
    pipe, grid = get_text_models()[name]
    start = time.time()
    search = GridSearchCV(pipe, grid, cv=3, scoring="f1_weighted", n_jobs=n_jobs)
    search.fit(data["X_train"], data["y_train"])
    pred = search.predict(data["X_test"])
    metrics = {"accuracy": float(accuracy_score(data["y_test"], pred)),
               "f1": float(f1_score(data["y_test"], pred, average="weighted"))}
    return {
        "model": name,
        "fitted": search.best_estimator_,
        "score": metrics["f1"],
        "metrics": metrics,
        "best_params": {k.replace("model__", ""): (v.item() if hasattr(v, "item") else v)
                        for k, v in search.best_params_.items()},
        "seconds": round(time.time() - start, 1),
    }


def text_worker(queue, name: str, data: dict):
    try:
        result = train_one_text(name, data, 1)
        result["fitted"] = pickle.dumps(result["fitted"])
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("error", str(e)))