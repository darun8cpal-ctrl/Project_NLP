Stock Head Tracker

A machine learning app that reads a financial news headline and identifies which stock it's talking about. Built using traditional NLP techniques — no transformers, no LLMs.

Live App


What it does

You paste in a headline like "Tesla faces supply chain disruptions at its Berlin Gigafactory" and the model returns the most likely stock ticker — in this case, $TSLA.

Under the hood it cleans the text, converts it to numerical features using Bag-of-Words and TF-IDF weighting, and runs it through a Multinomial Naive Bayes classifier trained on ~3,000 stock tickers.


Dataset

Kaggle Financial News Dataset — headlines labelled with their corresponding stock tickers, covering approximately 3,000 unique companies.


Performance

Tested on a held-out split of the dataset:

MetricScoreAccuracy42%Macro Precision73%Macro Recall39%Macro F146%

The gap between precision and recall comes down to class imbalance — many tickers appear very rarely in financial news, so the model is cautious but accurate when it does commit to a prediction.


Stack

Python, Scikit-learn, NLTK, Streamlit, yFinance, Joblib
