# Letterboxd Movie Recommendations

Get personalized film recommendations using your Letterboxd export and collaborative filtering (item-based) with MovieLens data.

## How it works

1. Export your ratings (and optionally liked films) from [Letterboxd](https://letterboxd.com).
2. Upload those CSV files to this app.
3. The app matches your films to MovieLens and generates recommendations via item-based collaborative filtering.

## Prerequisites

- Python 3.8 or newer
- pip

## Run locally

### 1. Download the project

```bash
git clone https://github.com/tiffpvrl/letterboxd-movie-recommendation.git
cd letterboxd-movie-recommendation
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the app

```bash
python app.py
```

### 5. Open in your browser

Go to [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Getting your Letterboxd data

1. Go to [letterboxd.com/settings/data](https://letterboxd.com/settings/data)
2. Click **Request my data**
3. Wait for the email with your download link
4. Download and extract the ZIP
5. Use `ratings.csv` (required) and optionally `likes/films.csv`

## Project structure

- `app.py` – Flask web app and API
- `lb_recs.py` – Letterboxd loading and MovieLens matching
- `collab_filter.py` – Item-based collaborative filtering
- `ml_data/` – MovieLens dataset (ratings.csv, movies.csv)
- `templates/` – Front-end HTML
