import sys 
import time 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import accuracy_score,f1_score,mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.svm import SVR,SVC

def detect_problem_type(y: pd.Series) -> str:
    if y.dtype == "object" or y.dtype == "bool" or str(y.dtype) == "category":
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= 15:
        return "classification"
    return "regression"
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
 
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])
def get_models(problem_type: str) -> dict:
    """Each entry: name -> (estimator, param_grid). Grids are kept small for speed."""
    if problem_type == "classification":
        return {
            "Logistic Regression": (
                LogisticRegression(max_iter=1000),
                {"model__C": [0.1, 1, 10]},
            ),
            "Random Forest": (
                RandomForestClassifier(random_state=42),
                {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]},
            ),
            "SVM": (
                SVC(),
                {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"]},
            ),
            "KNN": (
                KNeighborsClassifier(),
                {"model__n_neighbors": [3, 5, 9]},
            ),
            "Naive Bayes": (
                GaussianNB(),
                {"model__var_smoothing": [1e-9, 1e-8, 1e-7]},
            ),
        }
    return {
        "Ridge Regression": (
            Ridge(),
            {"model__alpha": [0.1, 1, 10]},
        ),
        "Random Forest": (
            RandomForestRegressor(random_state=42),
            {"model__n_estimators": [100, 200], "model__max_depth": [None, 10]},
        ),
        "SVR": (
            SVR(),
            {"model__C": [0.1, 1, 10], "model__kernel": ["rbf", "linear"]},
        ),
        "KNN": (
            KNeighborsRegressor(),
            {"model__n_neighbors": [3, 5, 9]},
        ),
    }
def run(csv_path: str, target: str | None = None):
    df = pd.read_csv(csv_path)
    target = target or df.columns[-1]
    if target not in df.columns:
        sys.exit(f"Target column '{target}' not found. Columns: {list(df.columns)}")
 
    df = df.dropna(subset=[target])
    X, y = df.drop(columns=[target]), df[target]
 
    problem_type = detect_problem_type(y)
    print(f"Rows: {len(df)} | Features: {X.shape[1]} | Target: {target} | Type: {problem_type}")
 
    stratify = None
    if problem_type == "classification" and y.value_counts().min() >= 2:
        stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify
    )
    scoring = "f1_weighted" if problem_type == "classification" else "r2"
    main_metric = "f1" if problem_type == "classification" else "r2"
    results = []
 
    for name, (estimator, grid) in get_models(problem_type).items():
        print(f"Training {name} ...", end=" ", flush=True)
        start = time.time()
        try:
            pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", estimator)])
            search = GridSearchCV(pipe, grid, cv=3, scoring=scoring, n_jobs=-1)
            search.fit(X_train, y_train)
            metrics = evaluate(problem_type, y_test, search.predict(X_test))
            results.append({
                "model": name,
                **metrics,
                "cv_score": search.best_score_,
                "best_params": {k.replace("model__", ""): v for k, v in search.best_params_.items()},
                "seconds": round(time.time() - start, 1),
            })
            print(f"done in {time.time() - start:.1f}s")
        except Exception as e:  # one model failing should not kill the whole run
            print(f"FAILED ({e})")
 
    if not results:
        sys.exit("All models failed.")
 
    board = pd.DataFrame(results).sort_values(main_metric, ascending=False).reset_index(drop=True)
    board.index += 1
    print("\n=== PODIUM LEADERBOARD ===")
    print(board.drop(columns=["best_params"]).to_string())
    winner = board.iloc[0]
    print(f"\nWinner: {winner['model']} ({main_metric} = {winner[main_metric]:.4f})")
    print(f"Best params: {winner['best_params']}")
    return board
 
 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python train.py <data.csv> [target_column]")
    run(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
 