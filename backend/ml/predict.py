import numpy as np
import pandas as pd


def _column_groups(pipe) -> dict:
    prep = pipe.named_steps["prep"]
    return {name: list(cols) for name, _, cols in prep.transformers_}


def feature_schema(pipe, df: pd.DataFrame) -> list:
    prep = pipe.named_steps["prep"]
    numeric = set(_column_groups(pipe).get("num", []))
    features = []
    for col in prep.feature_names_in_:
        if col in numeric:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                features.append({"name": col, "kind": "number",
                                 "default": round(float(s.median()), 3),
                                 "min": round(float(s.min()), 3),
                                 "max": round(float(s.max()), 3)})
            else:
                features.append({"name": col, "kind": "number", "default": 0})
        else:
            options = df[col].dropna().astype(str).value_counts().head(30).index.tolist()
            features.append({"name": col, "kind": "category",
                             "options": options,
                             "default": options[0] if options else ""})
    return features


def _native(value):
    return value.item() if hasattr(value, "item") else value


def run_prediction(pipe, row: dict, problem_type: str) -> dict:
    prep = pipe.named_steps["prep"]
    groups = _column_groups(pipe)

    clean = {}
    for col in prep.feature_names_in_:
        value = row.get(col)
        clean[col] = np.nan if value is None or value == "" else value
    X = pd.DataFrame([clean])
    for col in groups.get("num", []):
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in groups.get("cat", []):
        X[col] = X[col].astype(object)

    result = {"prediction": _native(pipe.predict(X)[0])}

    if problem_type == "classification" and hasattr(pipe, "predict_proba"):
        try:
            probs = pipe.predict_proba(X)[0]
            classes = pipe.named_steps["model"].classes_
            result["probabilities"] = {
                str(_native(c)): round(float(p), 4) for c, p in zip(classes, probs)
            }
        except Exception:
            pass
    return result