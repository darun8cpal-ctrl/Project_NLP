# Stock Head Tracker

An NLP-based machine learning application that classifies financial news headlines to their most relevant stock ticker. Built to explore how traditional Natural Language Processing techniques can extract meaningful signals from financial text data.

**[Live App](https://projectnlp-roep8upftaifwpsehkwyta.streamlit.app)**

---

## What It Does

Given a financial news headline as input, the model identifies which stock or asset the headline is most likely referring to and returns the corresponding ticker symbol along with the company name.

This is a **multi-class text classification problem** across approximately 3,000 stock tickers, trained on real financial news data.

---

## Dataset

- **Source:** Kaggle Financial News Dataset
- **Content:** Financial news headlines labelled with their corresponding stock tickers
- **Scale:** ~3,000 unique ticker classes
- **Preprocessing:** Lowercasing, punctuation removal, numeric removal, stopword filtering

```

---

## Model Performance

Evaluated on a held-out test set:

| Metric | Score |
|---|---|
| Accuracy | 42% |
| Macro Precision | 73% |
| Macro Recall | 39% |
| Macro F1 | 46% |

The high macro precision relative to recall reflects a class imbalance characteristic of financial news datasets — many tickers appear rarely, so the model is conservative but accurate when it does make a prediction. This is a known limitation and a direction for future improvement.

---

## Example

**Input:**
```
Tesla faces supply chain disruptions at its Berlin Gigafactory
```

**Output:**
```
$TSLA — Tesla, Inc.
```

---

## Project Structure

```
Project_NLP/
├── Project.ipynb                  # Full ML pipeline and evaluation
├── app.py                         # Streamlit frontend
├── stock_predictor_model.pkl      # Trained Naive Bayes model
├── bag_of_words_transformer.pkl   # Fitted CountVectorizer
├── tfidf_weight_transformer.pkl   # Fitted TF-IDF transformer
├── ticker_label_encoder.pkl       # Label encoder for ticker classes
└── requirements.txt
```

---

## Tech Stack

- **Language:** Python
- **ML & NLP:** Scikit-learn, NLTK
- **Data:** Pandas, NumPy
- **Deployment:** Streamlit, Joblib
- **Live Market Data:** yFinance

---

## Limitations & Future Work

- **Class imbalance** across 3,000 tickers affects recall for rare stocks
- **Bag-of-Words** loses word order and context — transformer-based models (e.g. FinBERT) would improve performance significantly
- **No sentiment signal** — a natural extension would be classifying bullish/bearish tone alongside ticker identification
- Future directions: confidence scoring, multi-label prediction, real-time news feed integration

---

## Notes

The ML pipeline, model training, and evaluation were built independently. The Streamlit frontend UI was developed with AI assistance (Gemini).
