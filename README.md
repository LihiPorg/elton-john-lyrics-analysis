# 🎶 Elton John Lyrics – NLP & Stylistic Analysis

A computational analysis of Elton John and Bernie Taupin’s lyrics across five decades using Natural Language Processing, stylometry, topic modeling, and machine learning classifiers.

## 📘 Project Overview

This project explores how Elton John’s lyrics evolved stylistically over time, with a focus on identifying linguistic changes across decades. We use a range of NLP tools to extract syntactic and lexical features from song lyrics, uncover hidden topics, and classify each song to its decade of origin.

## 🧪 Methods

- **Data Collection**: ~150 Elton John songs were scraped using the [Genius API](https://docs.genius.com/), representing multiple decades.
- **Morphological Analysis**: Extracted features such as:
  - Sentence length
  - Lexical diversity
  - Frequency of verbs, nouns, adjectives, and prepositions
  - Question sentence ratio
- **Topic Modeling**: Implemented Latent Dirichlet Allocation (LDA) using `gensim` to identify recurring lyrical themes.
- **Stylometry**: Applied statistical methods to track stylistic changes and correlate them with biographical events.
- **Decade Classification**:
  - Used `Logistic Regression` and `Random Forest` to predict the decade a song was written based on linguistic features.
  - Evaluation performed using Stratified K-Fold cross-validation and confusion matrix analysis.

## 🔍 Key Findings

- Strong stylistic shifts were identified in key years (e.g. 1979, 1989, 2004), aligning with major life or career events.
- LDA revealed two dominant emotional-topic clusters.
- Logistic Regression offered higher accuracy, while Random Forest better approximated the "correct" decade even when wrong.
- Classification accuracy was highest in decades with more data (1970s–1980s), and dropped in later decades due to data imbalance and stylistic convergence.

## ⚙️ Tech Stack

- Python
- scikit-learn
- Trankit (for NLP morphological parsing)
- Gensim (LDA)
- Matplotlib / Seaborn (visualizations)
- Pandas / NumPy

## 📈 Example Visualizations

- Sentence length over time
- Adjective usage spikes
- Preposition trends by decade
- Confusion matrices comparing classifier performance

## 🧠 Insights

> Lyrics reflect not just trends in music, but personal transformation. NLP allows us to map a songwriter's emotional and stylistic journey.


## ✍️ Authors

- Lihi Porgador
- Based on collaborative work analyzing Elton John & Bernie Taupin’s lyrical evolution


