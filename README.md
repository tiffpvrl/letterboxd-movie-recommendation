# Letterboxd Movie Recommendations

Get personalized film recommendations using your Letterboxd export and SVD collaborative filtering trained on MovieLens 32M.

## How it works

1. Export your ratings (and optionally liked films) from [Letterboxd](https://letterboxd.com).
2. Upload those CSV files to this app.
3. The app matches your films to MovieLens, projects your ratings into a latent-factor space (truncated SVD), and returns personalised recommendations.

## Prerequisites

- Python 3.11 or newer
- pip
- A free [Kaggle](https://www.kaggle.com) account (for automatic dataset download)

## Run locally

### 1. Download the project

```bash
git clone https://github.com/tiffpvrl/letterboxd-movie-recommendation.git
cd letterboxd-movie-recommendation
```

### 2. Create a virtual environment (recommended)

```bash
python3.13 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up Kaggle credentials

The app downloads MovieLens 32M automatically from Kaggle on first run. You need a (free) API token:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) and click **Create New Token**
2. Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`

Or run `python -c "import kagglehub; kagglehub.login()"` and paste your token.

### 5. Start the app

```bash
python app.py
```

On the first run, ML-32M is downloaded from Kaggle (~239 MB, cached afterwards) and the SVD model is trained and saved to `svd_model.pkl`. Subsequent starts load the cached model instantly.

### 6. Open in your browser

Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Getting your Letterboxd data

1. Go to [letterboxd.com/settings/data](https://letterboxd.com/settings/data)
2. Click **Request my data**
3. Wait for the email with your download link
4. Download and extract the ZIP
5. Use `ratings.csv` (required) and optionally `likes/films.csv`

## Project structure

- `app.py` — Flask web app and API
- `lb_recs.py` — Letterboxd loading and MovieLens matching
- `collab_filter.py` — SVD collaborative filtering with Kaggle/LensKit data loading
- `ml_data/` — MovieLens latest-small (legacy, kept for reference)
- `templates/` — Front-end HTML
