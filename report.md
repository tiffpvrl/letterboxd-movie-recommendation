# Film Recommendations from Letterboxd Data Using Truncated SVD

## Introduction

This project started with a simple question: can I turn my Letterboxd ratings into genuinely useful film recommendations?

Letterboxd is great for logging what you've watched, but its recommendation engine is limited. Meanwhile, the MovieLens dataset — maintained by GroupLens at the University of Minnesota — contains 32 million ratings from over 200,000 users across 87,000 films. That's a rich source of collective film taste.

The idea: take my personal Letterboxd ratings, embed them into the MovieLens rating ecosystem, and use collaborative filtering to surface films that people with similar taste have loved — but that I haven't seen yet.

---

## Data Sources

### Letterboxd Export

Letterboxd allows users to export their data as a ZIP file containing several CSVs. This project uses two:

- **ratings.csv** — every film the user has rated, with a 0.5–5.0 star rating
- **likes/films.csv** — films the user "liked" (hearted) without necessarily rating

Liked-but-unrated films are treated as implicit 5-star ratings, since a like signals strong positive sentiment. This expands the input signal for users who like more often than they rate.

### Why MovieLens?

The ideal dataset for this project would have been Letterboxd's own rating data — same platform, same user base, no title-matching headaches. But Letterboxd does not offer a public API. Access requires direct approval from their development team, and they do not grant keys for personal projects. Without API access, there's no way to query Letterboxd's rating database at scale.

MovieLens is the natural alternative: it's open-source, well-documented, and large enough to power meaningful collaborative filtering. The tradeoff is that bridging Letterboxd and MovieLens requires matching films by title and year — a fuzzy, lossy process — rather than joining on a shared identifier. Every unmatched film is a data point lost. This is a direct consequence of working across two separate ecosystems, and it motivates the title-normalisation pipeline described below.

### MovieLens 32M

MovieLens 32M is a stable benchmark dataset released in May 2024, containing:

| Metric | Value |
|--------|-------|
| Ratings | 32,000,000 |
| Users | 200,948 |
| Movies | 87,585 |
| Rating scale | 0.5 – 5.0 (half-star increments) |
| Timespan | 1995 – 2023 |

The dataset is loaded programmatically via **kagglehub** (Kaggle's Python API) and parsed through **LensKit**, a Python toolkit for recommender systems research. This avoids manual downloads — the data is fetched and cached on first run.

### Bridging the Two Datasets

Letterboxd and MovieLens use completely different identifiers. Letterboxd titles are plain text (e.g., "The Grand Budapest Hotel"), while MovieLens uses its own format with trailing articles and alternate titles:

```
Grand Budapest Hotel, The (2014)
Spirited Away (Sen to Chihiro no kamikakushi) (2001)
```

Matching is done by normalising titles — stripping punctuation, lowering case, removing leading/trailing articles ("The", "A", "An"), and extracting the primary title before any parenthesised foreign names. Each MovieLens title is indexed under multiple cleaned variants for O(1) lookup. Year matching with a ±1 tolerance handles minor discrepancies between sources.

---

## Method: Truncated SVD on the Rating Matrix

### Why Not Traditional Collaborative Filtering?

The classic approach to collaborative filtering — computing an item-item or user-user similarity matrix using cosine similarity — doesn't scale to MovieLens 32M. An item-item similarity matrix for 87,585 movies would require storing ~7.7 billion pairwise similarities. That's roughly 60 GB of floating-point numbers, which won't fit in memory on a typical machine.

Instead, this project uses **truncated singular value decomposition (SVD)**, a form of matrix factorization that compresses the rating data into a low-dimensional latent space.

### The Rating Matrix

The starting point is a sparse user-item matrix **R** of shape 200,948 × 87,585. Each entry R(u, m) holds user *u*'s rating for movie *m*, centered by subtracting the global mean rating (~3.5). Unrated entries are zero (absent in the sparse representation). Of the ~17.6 billion possible entries, only 32 million are filled — the matrix is 99.8% empty.

This matrix is stored as a **compressed sparse row (CSR)** matrix, which only keeps the non-zero entries in memory (~400 MB instead of the ~130 TB a dense matrix would require).

### Truncated SVD Decomposition

SVD factors the centered matrix into three components:

$$R \approx U \Sigma V^T$$

where:

- **U** (200,948 × *k*) — user factors: each user as a *k*-dimensional vector
- **Σ** (*k* × *k*) — diagonal matrix of singular values, capturing the importance of each factor
- **V^T** (*k* × 87,585) — item factors: each movie as a *k*-dimensional vector

With *k* = 50 factors, the decomposition compresses the original 17.6 billion-cell matrix into two small factor matrices totaling roughly 14.4 million values. This is what "truncated" means — we keep only the top 50 dimensions that explain the most variance in the data, discarding the rest as noise.

### What the Factors Represent

The 50 latent dimensions are not predefined. The SVD discovers them purely from the patterns of who rates what, and how. They often end up correlating with interpretable concepts:

- One factor might capture "mainstream blockbuster vs. arthouse"
- Another might capture "dark/serious vs. lighthearted"
- Another might separate "classic cinema vs. modern releases"

But they're not labelled — they're whatever directions in "taste space" best explain the variance in 32 million ratings.

Two films end up close together in this space not because they share metadata (genre, director, actors), but because **users who feel strongly about one tend to feel similarly about the other**. It's the correlation of ratings across users, not just co-occurrence, that drives similarity.

### Generating Recommendations for a New User

When a user uploads their Letterboxd data, the process is:

1. **Match** their rated films to MovieLens IDs via the title index
2. **Build a rating vector** — a sparse 87,585-dimensional vector with their centered ratings
3. **Project** into the 50-dimensional latent space:

$$\mathbf{u} = \mathbf{r} \cdot V$$

4. **Reconstruct** predicted scores for all movies:

$$\hat{\mathbf{r}} = \mathbf{u} \cdot V^T + \mu$$

5. **Rank** the unrated movies by predicted score and return the top *N*

The projection step (3) answers: "where does this user sit in taste-space?" The reconstruction (4) answers: "given that position, what would they probably rate every movie?" Films they've already rated are excluded from the results.

This is fast — the projection is a matrix-vector multiply of shape (1 × 87,585) × (87,585 × 50), and the reconstruction is (1 × 50) × (50 × 87,585). Both complete in milliseconds.

---

## Limitations

### The Cold Start Problem

The SVD only knows what's in the rating matrix. A film with very few ratings has almost no signal to learn from — its 50-dimensional vector is essentially noise. Very new or very obscure films won't be recommended well, regardless of their quality.

### No Content Awareness

The model has no concept of directors, actors, genres, themes, cinematography, or any other metadata. If two films by the same director have mostly non-overlapping audiences in MovieLens (one mainstream, one niche), the SVD may see them as unrelated — even if, to a film enthusiast, the connection is obvious.

For example: *Parasite* (2019) and *Memories of Murder* (2003) are both directed by Bong Joon-ho. If the populations of users who rated each film barely overlap, the model has no way to infer the connection. In practice, with 200K users this is often mitigated — enough "bridge users" exist to link well-known films. But the blind spot is real for less-watched work.

### Popularity Bias

Films with many ratings exert more influence on the latent factors. The model tends to recommend popular, well-rated films over hidden gems. This is inherent to collaborative filtering on aggregate data — popular films have more statistical signal, so the model is more confident about them.

### Title Matching is Imperfect

Despite normalisation and multi-variant indexing, some Letterboxd films fail to match their MovieLens counterpart. Differences in transliteration, regional titles, or missing entries mean a portion of the user's ratings are lost. Unmatched films are reported to the user but don't contribute to recommendations. This is a direct cost of not having access to Letterboxd's API — with a shared identifier, this entire problem disappears.

### Rating Granularity

Both Letterboxd and MovieLens use a 0.5–5.0 scale with half-star increments. Users can rate a film 3.5 or 4.0, but not 3.8. This quantisation compresses what might be a rich spectrum of opinion into ten discrete buckets. On Letterboxd in particular, users often compensate by writing reviews that convey nuance their star rating can't — a 4-star review might read very differently depending on whether the reviewer almost gave it 3.5 or nearly rounded up to 4.5. The SVD model sees only the number, never the text.

### Static Model

The SVD is trained once on a snapshot of MovieLens 32M (collected through October 2023). It doesn't update as new films are released or as user tastes shift. A film released in 2024 won't appear in recommendations.

---

## Possible Improvements

### Hybrid Approach

Combining collaborative filtering with **content-based features** (genres, directors, cast, plot keywords) would address the metadata blindness. A hybrid model could use the SVD signal where user overlap is strong and fall back to content similarity where it's weak.

### Implicit Feedback

Currently, only explicit ratings contribute. Incorporating implicit signals — watchlist additions, re-watches, time spent on a film's page — could enrich the model, especially for users who don't rate often.

### More Sophisticated Factorisation

Techniques like **Alternating Least Squares (ALS)** or **neural collaborative filtering** can model non-linear interactions and handle implicit feedback more naturally. LensKit provides an ALS implementation that could serve as a drop-in upgrade.

### Sentiment Analysis on Reviews

Letterboxd users frequently write reviews that carry signal beyond the star rating. A natural extension would be to run sentiment analysis on these reviews — using a pretrained language model (e.g., a fine-tuned BERT or a lighter model like VADER) — to extract a continuous sentiment score that supplements or adjusts the discrete rating. This would move the system into hybrid CF + NLP territory, capturing nuance that half-star increments miss. One challenge: Letterboxd reviews are often informal, sarcastic, or meme-heavy ("this movie broke me" could mean 5 stars or 1 star depending on context), which makes off-the-shelf sentiment models less reliable without domain-specific fine-tuning.

### Dynamic Retraining

Periodically retraining on updated MovieLens data (or incorporating user feedback in real time) would keep recommendations current.

---

## Conclusion

This project turns a personal Letterboxd export into personalised film recommendations by embedding the user's ratings into the latent structure of 32 million MovieLens ratings via truncated SVD. It's not a perfect system — it can't see metadata, struggles with obscure films, and skews popular — but it captures something real: the collective taste patterns of hundreds of thousands of moviegoers, compressed into 50 dimensions and applied to one person's viewing history.

The math is elegant (a matrix decomposition), the data is rich (32M ratings), and the result is practical: a web app that takes two CSV files and returns films worth watching.

---

## Technical Stack

| Component | Tool |
|-----------|------|
| Data loading | kagglehub, LensKit |
| Matrix factorisation | scikit-learn (TruncatedSVD), SciPy (sparse matrices) |
| Title matching | Custom normalisation with dictionary indexing |
| Web interface | Flask, vanilla HTML/JS |
| Language | Python 3.13 |

## References

- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems*, 5(4), 1–19.
- Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix Factorization Techniques for Recommender Systems. *IEEE Computer*, 42(8), 30–37.
- Halko, N., Martinsson, P. G., & Tropp, J. A. (2011). Finding Structure with Randomness: Probabilistic Algorithms for Constructing Approximate Matrix Decompositions. *SIAM Review*, 53(2), 217–288.
