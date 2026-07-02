# Detecting Fraudulent & Duplicate Reviews with NLP and Machine Learning

An end-to-end NLP pipeline that flags **fake and duplicate product reviews** by combining **Sentence-BERT** semantic embeddings, **cosine-similarity** duplicate detection, and a **Logistic Regression** classifier. The system distinguishes authentic reviews from fraudulent or near-duplicate ones to help restore trust in online product reviews, achieving **81% accuracy** with balanced performance across both classes.

---

## Motivation

Over 95% of shoppers read online reviews before buying, and most trust them as much as personal recommendations. That trust makes reviews a target: fraudulent and duplicate reviews — often lightly reworded copies of genuine ones — inflate ratings, mislead consumers, and are estimated to affect tens of billions in yearly consumer spending. Duplicate and near-duplicate content is a strong, scalable signal of this manipulation. This project asks whether **semantic similarity + supervised learning** can reliably separate real reviews from fake ones.

## Approach

The pipeline moves from raw review text to a real/fake prediction in six stages:

1. **Data preparation** — Clean ~40,000 Amazon product reviews across multiple categories: lowercasing, stopword and punctuation removal, and handling of missing values.
2. **Feature extraction (Sentence-BERT)** — Encode each review into a 768-dimensional semantic embedding using `all-MiniLM-L6-v2`, capturing meaning rather than just word overlap (e.g. *"Great product!"* and *"Amazing item!"* land close together).
3. **Duplicate detection (unsupervised)** — Compute pairwise cosine similarity between embeddings and flag near-duplicate reviews above a similarity threshold, producing a `duplicate` feature.
4. **Feature assembly** — Combine the BERT embeddings with the duplicate flag into a single feature set.
5. **Classification (supervised)** — Train a Logistic Regression classifier on an 80/20 stratified train/test split to predict *real* vs *fake*.
6. **Evaluation** — Assess with a classification report, confusion matrix, ROC/AUC, precision-recall, and learning curves; visualize class balance and word clouds for real vs fake reviews.

## Key Results

| Metric | Score |
|---|---|
| Accuracy | **0.81** |
| Precision (Fake / Real) | 0.81 / 0.81 |
| Recall (Fake / Real) | 0.80 / 0.81 |
| F1-score (Fake / Real) | 0.80 / 0.81 |

The model achieves **balanced precision, recall, and F1 (~0.81) across both classes**, showing it distinguishes fake from real reviews without favoring either class. This makes it a scalable, interpretable component that can complement existing fake-reviewer and fake-content detection systems.

## Tech Stack

`Python` · `pandas` · `NumPy` · `sentence-transformers (Sentence-BERT)` · `scikit-learn` · `Matplotlib` · `Seaborn` · `WordCloud`

## Repository Structure

```
.
├── Duplicate_reviews_detector_BERT.ipynb   # Main pipeline: BERT embeddings → duplicates → classifier → evaluation
├── experiments/
│   └── Duplicate_reviews_detector_NLP.ipynb   # Earlier TF-IDF baseline (no BERT)
├── report/
│   └── CSML_Report.pdf                      # Full write-up: methodology, results, references
├── data/                                    # Dataset not included — see note below
└── README.md
```

## Getting Started

```bash
# Install dependencies
pip install pandas numpy scikit-learn sentence-transformers matplotlib seaborn wordcloud
```

Then open `Duplicate_reviews_detector_BERT.ipynb` and run the cells in order. Update the dataset path at the top of the notebook to point to your local copy.

## Dataset

The project uses a publicly available fake-reviews dataset of Amazon product reviews (labeled real/fake, with product category and rating fields). The raw data is **not included** in this repository. Download it separately and place it in `data/`, then update the file path in the notebook.

## Future Work

- Replace Logistic Regression with stronger classifiers (e.g. gradient boosting, fine-tuned transformers) and compare.
- Fine-tune a transformer directly on the review text instead of using frozen embeddings.
- Add reviewer-behavior features (posting frequency, rating deviation) to complement textual signals.
- Validate on larger, more diverse, multi-language datasets.
- Tune the similarity threshold and hyperparameters, and test real-time scalability.

## Note

This is a research and academic project, not a production fraud-detection service. Any real-world deployment would require further validation, human oversight, and attention to fairness and privacy.

---

*Author: Deepali Dange · MS in Artificial Intelligence, University of Texas at Austin · Case Studies in Machine Learning*
