import os
from datetime import datetime
from pathlib import Path
from ml.train import (prepare_data, get_models, worker,
                      drop_noise_columns, detect_problem_type)
import pandas as pd
from fastapi.staticfiles import StaticFiles
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from database import SessionLocal, engine, Base
import models 
import multiprocessing as mp
import queue as queue_lib
import time
import joblib

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Podium")
@app.on_event("startup")
def cleanup_stuck_jobs():
    db = SessionLocal()
    stuck = db.query(models.Job).filter(models.Job.status.in_(["queued", "training"])).all()
    for job in stuck:
        job.status = "failed"
        job.current_task = "Failed"
        job.error = "Server restarted before this job finished"
    db.commit()
    db.close()

UPLOAD_DIR = "uploads"
MAX_UPLOAD_MB = 50  # isse bada CSV allow nahi
os.makedirs(UPLOAD_DIR, exist_ok=True)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.get("/")
def root():
    return {"status": "Podium backend is alive"}


@app.post("/upload")
def upload_csv(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Sirf CSV allow 
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are allowed")

    # 2. File ko uploads ho rhi jidr
    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    save_name = f"{stamp}_{os.path.basename(file.filename)}"
    save_path = os.path.join(UPLOAD_DIR, save_name)
    content = file.file.read()
    if len(content) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413,
                            detail=f"File is too large. Limit is {MAX_UPLOAD_MB} MB")
    with open(save_path, "wb") as f:
        f.write(content)

    # 3. pandas pdhke check kr rh shi csv h 
    try:
        df = pd.read_csv(save_path)
    except Exception:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail="Could not read this file as a CSV")
    if df.empty:
        os.remove(save_path)
        raise HTTPException(status_code=400, detail="The CSV is empty")

    # 4. datasets table mein ek row likh rh 
    dataset = models.Dataset(
        filename=file.filename,
        file_path=save_path,
        n_rows=len(df),
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    # 5. User ko output bej rh 
    return {
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "rows": len(df),
        "columns": list(df.columns),
    }


MAX_WORKERS = 8
MODEL_TIME_LIMIT = 300  # seconds: isse zyada lage to model band
MODEL_DIR = "models_saved"
os.makedirs(MODEL_DIR, exist_ok=True)


def count_trainings(problem_type: str, model_name: str) -> int:
    """GridSearchCV kitni baar train karega: settings x 3 folds."""
    grid = get_models(problem_type)[model_name][1]
    settings = 1
    for values in grid.values():
        settings *= len(values)
    return settings * 3


def run_jobs(dataset_id: int):
    """Background kaam: saare queued jobs ko ek saath train karta hai."""
    db = SessionLocal()  
    try:
        dataset = db.get(models.Dataset, dataset_id)
        data = prepare_data(dataset.file_path, dataset.target_column)
        waiting = (
            db.query(models.Job)
            .filter_by(dataset_id=dataset_id, status="queued")
            .order_by(models.Job.id)
            .all()
        )
        running = []
        best = {"score": None, "model": None, "job": None}
        ctx = mp.get_context("spawn")

        def start_next():
            """Line mein agla employee ho to use alag process mein kaam par lagao."""
            if not waiting:
                return
            job = waiting.pop(0)
            job.status = "training"
            job.started_at = datetime.utcnow()
            job.current_task = (
                f"GridSearchCV running: "
                f"{count_trainings(dataset.problem_type, job.model_name)} trainings"
            )
            db.add(models.BossLog(dataset_id=dataset_id,
                                  message=f"{job.model_name} started training."))
            db.commit()
            q = ctx.Queue()
            proc = ctx.Process(target=worker, args=(q, job.model_name, data))
            proc.start()
            running.append({"proc": proc, "queue": q, "job": job, "t0": time.time()})

        def check(item):
            """Result mile to (kind, payload), abhi chal raha ho to None."""
            try:
                return item["queue"].get_nowait()
            except queue_lib.Empty:
                pass
            if time.time() - item["t0"] > MODEL_TIME_LIMIT:
                item["proc"].terminate()
                return ("error", f"Time limit of {MODEL_TIME_LIMIT}s reached")
            if not item["proc"].is_alive():
                try:
                    return item["queue"].get(timeout=2)
                except queue_lib.Empty:
                    return ("error", "Training process crashed")
            return None

        def finish(job, kind, payload):
            if kind == "ok":
                job.status = "done"
                job.score = payload["score"]
                job.metrics = payload["metrics"]
                job.best_params = payload["best_params"]
                job.seconds = payload["seconds"]
                job.current_task = f"Finished in {payload['seconds']}s"
                if best["score"] is None or payload["score"] > best["score"]:
                    best.update(score=payload["score"], model=payload["model"], job=job)
                msg = f"{job.model_name} finished with score {payload['score']:.4f}."
            else:
                job.status = "failed"
                job.error = payload
                job.current_task = "Failed"
                msg = f"{job.model_name} failed: {payload}"
            job.finished_at = datetime.utcnow()
            db.add(models.BossLog(dataset_id=dataset_id, message=msg))
            db.commit()

        for _ in range(MAX_WORKERS):
            start_next()

        while running:
            time.sleep(0.5)
            for item in list(running):
                result = check(item)
                if result is None:
                    continue
                item["proc"].join()
                running.remove(item)
                finish(item["job"], *result)
                start_next()

        winner = (
            db.query(models.Job)
            .filter_by(dataset_id=dataset_id, status="done")
            .order_by(models.Job.score.desc())
            .first()
        )
        if winner:
            db.add(models.BossLog(
                dataset_id=dataset_id,
                message=f"Winner: {winner.model_name} with score {winner.score:.4f}.",
            ))
            db.commit()
        if best["model"] is not None:
            path = os.path.join(MODEL_DIR, f"dataset_{dataset_id}_{best['job'].model_name}.joblib")
            joblib.dump(best["model"], path)
            best["job"].model_path = path
            db.commit()
    finally:
        db.close()


@app.post("/run/{dataset_id}")
def run_dataset(dataset_id: int, target: str, background_tasks: BackgroundTasks,
                db: Session = Depends(get_db)):
    # 1. Dataset dhundhege 
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # 2. Data tayyar kr rhe
    try:
        data = prepare_data(dataset.file_path, target)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 3. Dataset ki misssing detail bhr rhe 
    dataset.target_column = target
    dataset.problem_type = data["problem_type"]
    dataset.n_features = data["n_features"]

    # 4. Har model ke liye ek job (employee) ban rh, status "queued"
    model_names = list(get_models(data["problem_type"]))
    jobs = []
    for name in model_names:
        job = models.Job(dataset_id=dataset.id, model_name=name, status="queued",
                         current_task="Waiting in line")
        db.add(job)
        jobs.append(job)

    # 5. Boss ka message
    db.add(models.BossLog(
        dataset_id=dataset.id,
        message=f"Boss: new {data['problem_type']} project '{dataset.filename}' "
                f"(target: {target}). {len(jobs)} employees assigned.",
    ))
    db.commit()

    # 6. Training background mein shuru karo, user ko turant jawab do
    background_tasks.add_task(run_jobs, dataset.id)

    return {
        "dataset_id": dataset.id,
        "problem_type": data["problem_type"],
        "jobs": [{"job_id": j.id, "model": j.model_name, "status": j.status} for j in jobs],
    }

@app.get("/jobs/{dataset_id}")
def get_jobs(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    jobs = (
        db.query(models.Job)
        .filter_by(dataset_id=dataset_id)
        .order_by(models.Job.id)
        .all()
    )
    return [
        {
            "job_id": j.id,
            "model": j.model_name,
            "status": j.status,
            "current_task": j.current_task,
            "started_at": j.started_at,
            "score": j.score,
            "metrics": j.metrics,
            "best_params": j.best_params,
            "seconds": j.seconds,
            "error": j.error,
        }
        for j in jobs
    ]
@app.get("/leaderboard/{dataset_id}")
def get_leaderboard(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    done_jobs = (
        db.query(models.Job)
        .filter_by(dataset_id=dataset_id, status="done")
        .order_by(models.Job.score.desc())
        .all()
    )
    return [
        {
            "rank": rank,
            "model": j.model_name,
            "score": j.score,
            "metrics": j.metrics,
            "seconds": j.seconds,
        }
        for rank, j in enumerate(done_jobs, start=1)
    ]
@app.get("/logs/{dataset_id}")
def get_logs(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    logs = (
        db.query(models.BossLog)
        .filter_by(dataset_id=dataset_id)
        .order_by(models.BossLog.id)
        .all()
    )
    return [
        {"id": l.id, "message": l.message, "time": l.created_at}
        for l in logs
    ]


@app.get("/insights/{dataset_id}")
def get_insights(dataset_id: int, target: str | None = None,
                 db: Session = Depends(get_db)):
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    df = pd.read_csv(dataset.file_path)
    target = target or dataset.target_column
    if target is not None and target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column '{target}' not found")

    features = df.drop(columns=[target]) if target else df
    _, dropped = drop_noise_columns(features)
    n_numeric = len(features.select_dtypes(include="number").columns)

    # Kin columns mein khali (missing) values hain
    missing = []
    for col, count in df.isna().sum().items():
        if count > 0:
            missing.append({"column": col, "missing": int(count),
                            "percent": round(count / len(df) * 100, 1)})

    result = {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric_features": n_numeric,
        "text_features": len(features.columns) - n_numeric,
        "duplicate_rows": int(df.duplicated().sum()),
        "missing": missing,
        "dropped_columns": dropped,
        "target": None,
        "warnings": [],
    }

    # Target column ki jaankari
    if target:
        y = df[target].dropna()
        problem_type = detect_problem_type(y)
        info = {"column": target, "type": problem_type}
        if problem_type == "classification":
            share = (y.value_counts(normalize=True) * 100).round(1).head(10)
            info["balance_percent"] = {str(k): float(v) for k, v in share.items()}
            if share.min() < 10:
                result["warnings"].append(
                    "Target classes are imbalanced: a small class has under 10% of rows.")
        else:
            info.update({"min": float(y.min()), "max": float(y.max()),
                         "mean": round(float(y.mean()), 3),
                         "median": round(float(y.median()), 3)})
        result["target"] = info

    # Seedhi bhasha mein chetavni
    for item in missing:
        if item["percent"] > 30:
            result["warnings"].append(
                f"Column '{item['column']}' is {item['percent']}% empty.")
    if len(df) < 500:
        result["warnings"].append("Small dataset (under 500 rows): scores can vary a lot.")
    if dropped:
        result["warnings"].append(
            "These columns were ignored as noise: " + ", ".join(dropped) + ".")

    return result


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/app", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")