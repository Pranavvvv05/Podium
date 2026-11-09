import os
from datetime import datetime
from pathlib import Path
from ml.train import (prepare_data, get_models, worker,
                      drop_noise_columns, detect_problem_type)
import pandas as pd
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from database import SessionLocal, engine, Base
import models 
import multiprocessing as mp
import queue as queue_lib
import time
import joblib
from pydantic import BaseModel
from ml.predict import feature_schema, run_prediction
from ml.vision import (extract_zip, scan_folder, prepare_image_data,
                       get_image_models, image_worker, load_image, IMG_SIZE)
from ml.nlp import (write_text_csv, prepare_text_data, get_text_models, text_worker)
import pickle
import shutil
import zipfile

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
DEEP_TIME_LIMIT = 900  # ANN/CNN/RNN/TextCNN CPU par zyada time lete hain
DEEP_MODELS = {"ANN", "CNN", "RNN", "TextCNN"}


def time_limit_for(model_name: str) -> int:
    return DEEP_TIME_LIMIT if model_name in DEEP_MODELS else MODEL_TIME_LIMIT
MODEL_DIR = "models_saved"
os.makedirs(MODEL_DIR, exist_ok=True)


IMAGE_TYPE = "image_classification"
MAX_ZIP_MB = 200


TEXT_TYPE = "text_classification"
DEEP_TYPES = (IMAGE_TYPE, TEXT_TYPE)


def text_data_path(dataset) -> str:
    return dataset.file_path + ".text.csv"


def models_for(problem_type: str) -> dict:
    if problem_type == IMAGE_TYPE:
        return get_image_models()
    if problem_type == TEXT_TYPE:
        return get_text_models()
    return get_models(problem_type)


def count_trainings(problem_type: str, model_name: str) -> int:
    """GridSearchCV kitni baar train karega: settings x 3 folds."""
    grid = models_for(problem_type)[model_name][1]
    settings = 1
    for values in grid.values():
        settings *= len(values)
    return settings * 3


def run_jobs(dataset_id: int):
    """Background kaam: saare queued jobs ko ek saath train karta hai."""
    db = SessionLocal()  
    try:
        dataset = db.get(models.Dataset, dataset_id)
        if dataset.problem_type == IMAGE_TYPE:
            data = prepare_image_data(dataset.file_path)
            work = image_worker
        elif dataset.problem_type == TEXT_TYPE:
            data = prepare_text_data(text_data_path(dataset))
            work = text_worker
        else:
            data = prepare_data(dataset.file_path, dataset.target_column)
            work = worker
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
            proc = ctx.Process(target=work, args=(q, job.model_name, data))
            proc.start()
            running.append({"proc": proc, "queue": q, "job": job, "t0": time.time()})

        def check(item):
            """Result mile to (kind, payload), abhi chal raha ho to None."""
            try:
                return item["queue"].get_nowait()
            except queue_lib.Empty:
                pass
            limit = time_limit_for(item["job"].model_name)
            if time.time() - item["t0"] > limit:
                item["proc"].terminate()
                return ("error", f"Time limit of {limit}s reached")
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
                    best.update(score=payload["score"], model=pickle.loads(payload["fitted"]), job=job)
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

@app.post("/upload-images")
def upload_images(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are allowed")

    stamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    zip_path = os.path.join(UPLOAD_DIR, f"{stamp}.zip")
    folder = os.path.join(UPLOAD_DIR, f"images_{stamp}")
    content = file.file.read()
    if len(content) > MAX_ZIP_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File is too large. Limit is {MAX_ZIP_MB} MB")
    with open(zip_path, "wb") as f:
        f.write(content)

    try:
        classes = extract_zip(zip_path, folder)
    except zipfile.BadZipFile:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Could not read this file as a ZIP")
    except ValueError as e:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        os.remove(zip_path)

    total = sum(len(v) for v in classes.values())
    dataset = models.Dataset(filename=file.filename, file_path=folder, n_rows=total,
                             target_column="folder name", problem_type=IMAGE_TYPE,
                             n_features=IMG_SIZE * IMG_SIZE * 3)
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return {
        "dataset_id": dataset.id,
        "filename": dataset.filename,
        "images": total,
        "classes": {name: len(files) for name, files in classes.items()},
    }


@app.post("/run-images/{dataset_id}")
def run_image_dataset(dataset_id: int, background_tasks: BackgroundTasks,
                      db: Session = Depends(get_db)):
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None or dataset.problem_type != IMAGE_TYPE:
        raise HTTPException(status_code=404, detail="Image dataset not found")

    jobs = []
    for name in get_image_models():
        job = models.Job(dataset_id=dataset.id, model_name=name, status="queued",
                         current_task="Waiting in line")
        db.add(job)
        jobs.append(job)
    db.add(models.BossLog(
        dataset_id=dataset.id,
        message=f"Boss: new image project '{dataset.filename}' "
                f"({dataset.n_rows} images). {len(jobs)} employees assigned.",
    ))
    db.commit()
    background_tasks.add_task(run_jobs, dataset.id)
    return {
        "dataset_id": dataset.id,
        "problem_type": IMAGE_TYPE,
        "jobs": [{"job_id": j.id, "model": j.model_name, "status": j.status} for j in jobs],
    }


@app.post("/run-text/{dataset_id}")
def run_text_dataset(dataset_id: int, text_column: str, label_column: str,
                     background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Use a CSV uploaded with /upload: pick the text column and the label column."""
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None or dataset.problem_type == IMAGE_TYPE:
        raise HTTPException(status_code=404, detail="CSV dataset not found")
    try:
        counts = write_text_csv(dataset.file_path, text_column, label_column,
                                dataset.file_path + ".text.csv")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    dataset.target_column = label_column
    dataset.problem_type = TEXT_TYPE
    dataset.n_features = 1
    jobs = []
    for name in get_text_models():
        job = models.Job(dataset_id=dataset.id, model_name=name, status="queued",
                         current_task="Waiting in line")
        db.add(job)
        jobs.append(job)
    db.add(models.BossLog(
        dataset_id=dataset.id,
        message=f"Boss: new text project '{dataset.filename}' (text: {text_column}, "
                f"label: {label_column}, {len(counts)} classes). {len(jobs)} employees assigned.",
    ))
    db.commit()
    background_tasks.add_task(run_jobs, dataset.id)
    return {
        "dataset_id": dataset.id,
        "problem_type": TEXT_TYPE,
        "classes": counts,
        "jobs": [{"job_id": j.id, "model": j.model_name, "status": j.status} for j in jobs],
    }


def latest_run_jobs(db: Session, dataset) -> list:
    """Sirf sabse naye run ki jobs (purane runs ki jobs nahi)."""
    query = db.query(models.Job).filter_by(dataset_id=dataset.id)
    if not dataset.problem_type:
        return query.order_by(models.Job.id).all()
    n = len(models_for(dataset.problem_type))
    recent = query.order_by(models.Job.id.desc()).limit(n).all()
    return sorted(recent, key=lambda j: j.id)


@app.get("/jobs/{dataset_id}")
def get_jobs(dataset_id: int, db: Session = Depends(get_db)):
    dataset = db.get(models.Dataset, dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="Dataset not found")

    jobs = latest_run_jobs(db, dataset)
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

    done_jobs = sorted(
        [j for j in latest_run_jobs(db, dataset) if j.status == "done"],
        key=lambda j: j.score,
        reverse=True,
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

    if dataset.problem_type == IMAGE_TYPE:
        classes = scan_folder(dataset.file_path)
        total = sum(len(v) for v in classes.values())
        return {
            "images": total,
            "classes": {k: len(v) for k, v in classes.items()},
            "balance_percent": {k: round(len(v) / total * 100, 1) for k, v in classes.items()},
            "warnings": ["Small dataset (under 300 images): scores can vary a lot."] if total < 300 else [],
        }

    if dataset.problem_type == TEXT_TYPE:
        tdf = pd.read_csv(text_data_path(dataset))
        share = (tdf["label"].value_counts(normalize=True) * 100).round(1)
        return {
            "rows": len(tdf),
            "classes": {str(k): int(v) for k, v in tdf["label"].value_counts().items()},
            "balance_percent": {str(k): float(v) for k, v in share.items()},
            "avg_words": round(float(tdf["text"].str.split().str.len().mean()), 1),
            "warnings": ["Small dataset (under 500 rows): scores can vary a lot."] if len(tdf) < 500 else [],
        }

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

def latest_winner(db: Session, dataset_id: int):
    """Winning job of the newest run that has a saved model file."""
    winner = (
        db.query(models.Job)
        .filter(models.Job.dataset_id == dataset_id,
                models.Job.status == "done",
                models.Job.model_path.isnot(None))
        .order_by(models.Job.id.desc())
        .first()
    )
    if not winner or not os.path.exists(winner.model_path):
        raise HTTPException(status_code=404, detail="Saved .joblib model not found on server")
    return winner


@app.get("/download/{dataset_id}")
def download_winner_model(dataset_id: int, db: Session = Depends(get_db)):
    winner = latest_winner(db, dataset_id)
    return FileResponse(
        path=winner.model_path,
        filename=os.path.basename(winner.model_path),
        media_type="application/octet-stream"
    )


@app.get("/features/{dataset_id}")
def get_features(dataset_id: int, db: Session = Depends(get_db)):
    """Which inputs the winning model needs (used to build the prediction form)."""
    winner = latest_winner(db, dataset_id)
    dataset = db.get(models.Dataset, dataset_id)
    if dataset.problem_type in DEEP_TYPES:
        raise HTTPException(status_code=400, detail="This is an image/text model: use /predict-image or /predict-text")
    pipe = joblib.load(winner.model_path)
    return {
        "model": winner.model_name,
        "score": winner.score,
        "problem_type": dataset.problem_type,
        "target": dataset.target_column,
        "features": feature_schema(pipe, pd.read_csv(dataset.file_path)),
    }


class PredictRequest(BaseModel):
    row: dict


@app.post("/predict/{dataset_id}")
def predict(dataset_id: int, req: PredictRequest, db: Session = Depends(get_db)):
    """Real prediction from the winning model for one row of input values."""
    winner = latest_winner(db, dataset_id)
    dataset = db.get(models.Dataset, dataset_id)
    if dataset.problem_type in DEEP_TYPES:
        raise HTTPException(status_code=400, detail="This is an image/text model: use /predict-image or /predict-text")
    pipe = joblib.load(winner.model_path)
    try:
        result = run_prediction(pipe, req.row, dataset.problem_type)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not predict: {e}")
    return {
        "model": winner.model_name,
        "score": winner.score,
        "problem_type": dataset.problem_type,
        "target": dataset.target_column,
        **result,
    }


@app.post("/predict-image/{dataset_id}")
def predict_image(dataset_id: int, file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Real prediction from the winning image model for one uploaded picture."""
    winner = latest_winner(db, dataset_id)
    dataset = db.get(models.Dataset, dataset_id)
    if dataset.problem_type != IMAGE_TYPE:
        raise HTTPException(status_code=400, detail="This is not an image model: use /predict")
    try:
        x = load_image(file.file)[None]
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this file as an image")
    pipe = joblib.load(winner.model_path)
    proba = pipe.predict_proba(x)[0]
    classes = [str(c) for c in pipe.classes_]
    best = int(proba.argmax())
    return {
        "model": winner.model_name,
        "score": winner.score,
        "problem_type": IMAGE_TYPE,
        "prediction": classes[best],
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
    }


class TextRequest(BaseModel):
    text: str


@app.post("/predict-text/{dataset_id}")
def predict_text(dataset_id: int, req: TextRequest, db: Session = Depends(get_db)):
    """Real prediction from the winning text model for one sentence."""
    winner = latest_winner(db, dataset_id)
    dataset = db.get(models.Dataset, dataset_id)
    if dataset.problem_type != TEXT_TYPE:
        raise HTTPException(status_code=400, detail="This is not a text model")
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty")
    pipe = joblib.load(winner.model_path)
    proba = pipe.predict_proba([req.text])[0]
    classes = [str(c) for c in pipe.classes_]
    best = int(proba.argmax())
    return {
        "model": winner.model_name,
        "score": winner.score,
        "problem_type": TEXT_TYPE,
        "prediction": classes[best],
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
    }