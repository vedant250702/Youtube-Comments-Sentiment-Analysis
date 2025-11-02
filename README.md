# YouTube Comments Sentiment Analysis  

This project aims to analyze and classify YouTube comments into **positive**, **negative**, or **neutral** sentiments using various **Machine Learning algorithms**.  
It helps content creators and marketers understand audience feedback efficiently without manual comment reading.

---

## Problem Statement  

YouTube comment sections contain valuable feedback but are **unstructured and vast**, making manual analysis inefficient.  
This project builds a **sentiment classification model** to automatically determine viewer sentiments and enhance engagement insights.

---

## Objectives  

1. Analyze the sentiments of YouTube comments as **Positive**, **Negative**, or **Neutral**.  
2. Preprocess data using **language filtering**, **stopwords removal**, **lemmatization**, **tokenization**, and **emoji handling**.  
3. Compare performance of multiple ML models and choose the best one based on evaluation metrics.

---

## Dataset  

The project uses two Kaggle datasets combined for better variation and balance:

- [YouTube Comments Dataset (18K datapoints)](https://www.kaggle.com/datasets/atifaliak/youtube-comments-dataset)  
- [YouTube Comments Sentiment Dataset (1M datapoints)](https://www.kaggle.com/datasets/amaanpoonawala/youtube-comments-sentiment-dataset)  

After preprocessing and language filtering, around **880,000 English comments** were retained for model training.

---

## Data Preprocessing  

| Step | Description |
|------|--------------|
| **Language Detection** | Retained only English comments using `langdetect`. |
| **Stopwords Removal** | Removed irrelevant words while keeping sentiment-related terms (`not`, `no`, `very`, etc.). |
| **Contraction Expansion** | Expanded short forms (e.g., *can’t → cannot*). |
| **Named Entity Removal** | Removed names, places, and organizations to reduce bias. |
| **Tokenization & Lemmatization** | Split text into tokens and reduced to base forms. |
| **Punctuation Removal** | Removed symbols for cleaner vocabulary. |
| **Emoji Handling** | Converted emojis to descriptive words (e.g., 👍 → “:thumbs_up:”). |

---

## Feature Extraction  

- Used **TF-IDF Vectorization** to convert text into numerical form.  
- **ngram_range = (1,3)** tested for unigrams, bigrams, and trigrams.  
- Helps highlight important words and reduce the influence of common terms.

---

## Machine Learning Models  

| Model | Accuracy | Highlights |
|--------|-----------|------------|
| **Logistic Regression** | ~70% | Balanced performance across all sentiment classes. |
| **Naïve Bayes** | ~70% | Excellent recall for positive and negative classes. |
| **Decision Tree** | ~40% | Poor performance, prone to overfitting. |
| **Gradient Boosting** | ~68% | Strong for positive and negative detection. |
| **XGBoost** | ~63% | Underperformed due to limited computational tuning. |

---

## Key Findings  

- **Best Overall Model:** Logistic Regression – balanced and stable.  
- **Fastest Model:** Naïve Bayes – quick and effective for large data.  
- **Boosting Models:** Could outperform others with further tuning and GPU usage.  
- **Decision Tree:** Not suitable for large datasets due to overfitting.  
- **Future Work:** Use pre-trained embeddings like **Word2Vec** or **BERT** for improved context and sarcasm detection.


## Tech Stack  

- **Python**  
- **Pandas**, **NumPy**, **Scikit-learn**  
- **NLTK**, **Langdetect**  
- **Matplotlib**, **Seaborn** (for visualization)  

---
