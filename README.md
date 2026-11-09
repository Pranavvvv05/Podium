# Podium: AutoML Office

Podium is an AutoML platform where models compete like employees in a multi-floor pixel-art office. Upload a dataset, and many models train in parallel, tuned with GridSearchCV. A live leaderboard ranks them, and the winner stands on the podium and can be used for real predictions.

## What it does

- **Three input types**
  - Tabular CSV (classification or regression)
  - Image ZIP (one folder per class)
  - Text CSV (choose the text column and the label column)
- **Floor 1: Classical ML** (scikit-learn)
  - Classification: Logistic Regression, Random Forest, SVM, KNN, Naive Bayes
  - Regression: Ridge, Random Forest, SVR, KNN
- **Floor 2: Deep Learning** (PyTorch)
  - Tabular: ANN, CNN (1D), RNN (GRU)
  - Images: ANN, CNN (2D)
  - Text: ANN (TF-IDF), RNN (bi-GRU), TextCNN
- **Parallel training**: every model runs in its own process with a time limit (300 s classical, 900 s deep).
- **Live leaderboard** with scores, best parameters and training time.
- **Prediction Sandbox**: use the winning model on new input (form values, an image, or a sentence) and get class probabilities.
- **Download** the winning model as a fitted pipeline (joblib).
- **Insights**: feature importance for tabular data, class summary for images and text.

## Tech stack

FastAPI, PostgreSQL (SQLAlchemy), scikit-learn, PyTorch, Pillow, plain HTML/CSS/JS frontend.

## Architecture

```
Browser (HTML/JS UI at /app/)
        |  REST + polling every 0.5 s
FastAPI (main.py)
   |-- upload / run endpoints (BackgroundTasks)
   |-- one spawned process + queue per model
   |       ml/train.py   tabular models
   |       ml/vision.py  image models
   |       ml/nlp.py     text models
   |       ml/deep.py    PyTorch models (sklearn-compatible wrappers)
   |-- PostgreSQL: datasets, jobs, boss log
```

The PyTorch models are wrapped as scikit-learn estimators, so they work inside `Pipeline`, `GridSearchCV` and `joblib` like any other model.

## Setup

1. Create a PostgreSQL database named `podium_db` and set the connection details in `backend/database.py`.
2. Create and activate a virtual environment.
   ```
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r backend/requirements.txt
   ```
3. Run the server from the `backend` folder.
   ```
   cd backend
   uvicorn main:app --reload
   ```
4. Open https://podium-1-5gku.onrender.com

## Try it with sample data

```
cd backend
python make_test_data.py
```

This creates `test_data/digits_images.zip` (handwritten digits) and `test_data/news_text.csv` (4 newsgroup categories).

| Mode | File | Settings |
|---|---|---|
| Vision ZIP | `digits_images.zip` | none |
| NLP Text | `news_text.csv` | text column `text`, label column `category` |
| Tabular | any CSV | choose the target column |

## Results (sample runs)

| Dataset | Task | Best model | Score |
|---|---|---|---|
| Digits (1,500 images, 32x32) | Image classification | CNN | 97.4% accuracy |
| Digits (1,500 images, 32x32) | Image classification | ANN | 95.7% accuracy |
| 20 Newsgroups (4 classes) | Text classification | _fill in_ | _fill in_ |
| Heart Disease (918 rows, tabular) | Classification | KNN (Ned-KNN) | 91.8% accuracy |

## Screenshots

**Floor 1: Classical ML Bullpen**
<img width="1917" height="1078" alt="floor1" src="https://github.com/user-attachments/assets/105c34b2-af36-4683-9876-2204f4b7e631" />


**Floor 2: Deep Learning Lab**

<img width="1917" height="1077" alt="floor2" src="https://github.com/user-attachments/assets/ec0f116b-eb25-497b-9509-1db6c9badbbe" />


**Live leaderboard and podium**

<img width="575" height="792" alt="leaderboard" src="https://github.com/user-attachments/assets/62cf52b5-2175-4441-9404-eec742d4d0ed" />


**Boss logs (live training events)**

<img width="590" height="802" alt="bosslogs" src="https://github.com/user-attachments/assets/822a4e2b-b2bf-46e6-99d0-f15d76af15c7" />


**Prediction Sandbox**

![Sandbox](docs/sandbox.png)

## API overview

| Endpoint | Purpose |
|---|---|
| `POST /upload` | Upload a tabular CSV |
| `POST /upload-images` | Upload an image ZIP |
| `POST /run/{id}` | Train tabular models |
| `POST /run-images/{id}` | Train image models |
| `POST /run-text/{id}` | Train text models (`text_column`, `label_column`) |
| `GET /jobs/{id}`, `GET /leaderboard/{id}` | Progress and ranking |
| `GET /features/{id}`, `POST /predict/{id}` | Tabular prediction |
| `POST /predict-image/{id}` | Image prediction |
| `POST /predict-text/{id}` | Text prediction |
| `GET /insights/{id}` | Insights for the winner |

## Notes

- Images are resized to 32x32 RGB. Up to 6,000 images, at least 10 per class.
- Text datasets are capped at 20,000 rows, with at least 10 rows per label.
- Torch is optional for the tabular floor. Without it, a scikit-learn MLP is used instead.
