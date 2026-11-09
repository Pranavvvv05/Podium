import os
import pickle
import zipfile
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import GridSearchCV
import time

try:
    from ml.deep import TorchImageClassifier
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

IMG_SIZE = 32
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MAX_IMAGES = 6000
MIN_PER_CLASS = 10


def flatten_scale(X):
    return np.asarray(X, dtype=np.float32).reshape(len(X), -1) / 255.0


def load_image(source) -> np.ndarray:
    img = Image.open(source).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.uint8)


def extract_zip(zip_path: str, out_dir: str) -> dict:
    """Safely unzip image files. Returns {class_name: [file paths]}."""
    classes = {}
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.infolist()
                   if not m.is_dir() and m.filename.lower().endswith(IMG_EXT)
                   and "__MACOSX" not in m.filename]
        if len(members) > MAX_IMAGES:
            raise ValueError(f"Too many images ({len(members)}). Limit is {MAX_IMAGES}")
        for i, m in enumerate(members):
            parts = [p for p in m.filename.replace("\\", "/").split("/") if p]
            if len(parts) < 2 or ".." in parts:
                continue
            label = parts[-2]
            dest_dir = os.path.join(out_dir, label)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, f"{i}_{os.path.basename(parts[-1])}")
            with zf.open(m) as src, open(dest, "wb") as dst:
                dst.write(src.read())
            classes.setdefault(label, []).append(dest)
    if len(classes) < 2:
        raise ValueError("Need at least 2 class folders, each containing images")
    small = [c for c, files in classes.items() if len(files) < MIN_PER_CLASS]
    if small:
        raise ValueError(f"Each class needs at least {MIN_PER_CLASS} images. Too few in: {', '.join(small)}")
    return classes


def scan_folder(folder: str) -> dict:
    classes = {}
    for label in sorted(os.listdir(folder)):
        d = os.path.join(folder, label)
        if os.path.isdir(d):
            classes[label] = [os.path.join(d, f) for f in sorted(os.listdir(d))]
    return classes


def prepare_image_data(folder: str) -> dict:
    X, y = [], []
    for label, files in scan_folder(folder).items():
        for path in files:
            try:
                X.append(load_image(path))
                y.append(label)
            except Exception:
                continue
    X, y = np.stack(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    return {"problem_type": "image_classification", "n_rows": len(X),
            "n_features": IMG_SIZE * IMG_SIZE * 3,
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


def get_image_models() -> dict:
    models = {
        "ANN": (
            Pipeline([("flat", FunctionTransformer(flatten_scale)),
                      ("model", MLPClassifier(max_iter=200, early_stopping=True, random_state=42))]),
            {"model__hidden_layer_sizes": [(128,), (128, 64)], "model__alpha": [1e-4, 1e-2]},
        ),
    }
    if HAS_TORCH:
        models["CNN"] = (
            Pipeline([("model", TorchImageClassifier())]),
            {"model__lr": [0.003]},
        )
    return models


def train_one_image(name: str, data: dict, n_jobs: int = 1) -> dict:
    pipe, grid = get_image_models()[name]
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


def image_worker(queue, name: str, data: dict):
    try:
        result = train_one_image(name, data, 1)
        result["fitted"] = pickle.dumps(result["fitted"])
        queue.put(("ok", result))
    except Exception as e:
        queue.put(("error", str(e)))