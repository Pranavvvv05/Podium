import os
from datetime import datetime

import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from database import SessionLocal, engine, Base
import models 

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Podium")

UPLOAD_DIR = "uploads"
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
    with open(save_path, "wb") as f:
        f.write(file.file.read())

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