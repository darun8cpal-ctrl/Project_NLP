# Stock Head Tracker

Stock Head Tracker is an NLP-based machine learning application that predicts the most relevant stock ticker from a financial news headline.

The project was built to explore how traditional Natural Language Processing techniques can be used to extract meaningful information from financial text data and perform real-time classification.

## Features

* Financial headline classification
* Text preprocessing and tokenization
* Stopword removal and cleaning
* Bag-of-Words feature extraction
* TF-IDF weighting
* Multinomial Naive Bayes classification
* Streamlit web interface
* Real-time prediction and company lookup

## Project Workflow

1. Collect and prepare financial headline data.
2. Clean and preprocess text.
3. Convert text into numerical features using Bag-of-Words.
4. Apply TF-IDF weighting.
5. Train a Multinomial Naive Bayes classifier.
6. Encode stock ticker labels.
7. Save trained artifacts using Joblib.
8. Deploy the complete pipeline using Streamlit.

## Technologies Used

### Programming Language

* Python

### Libraries

* Scikit-Learn
* NLTK
* Streamlit
* Pandas
* NumPy
* Joblib
* yFinance

### Machine Learning Techniques

* Natural Language Processing (NLP)
* Bag-of-Words Vectorization
* TF-IDF Transformation
* Multinomial Naive Bayes Classification
* Label Encoding

## Example Use Case

Input Headline:

"Microsoft Office sales surge unexpectedly across Asian markets"

Predicted Output:

ORC (ORACLE Inc.)

## Future Improvements

* Financial sentiment analysis
* Transformer-based NLP models
* Confidence scoring
* News aggregation pipeline
* Multi-label prediction
* Real-time market data integration

## Learning Outcomes

Through this project I gained hands-on experience with:

* Building complete NLP pipelines
* Text preprocessing and feature engineering
* Machine learning model training
* Model serialization and deployment
* Creating interactive ML applications with Streamlit


