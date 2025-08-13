import os
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora, models
from gensim.models import CoherenceModel

def main():
    nltk.download("punkt")
    nltk.download("stopwords")
    nltk.download("wordnet")

    LYRICS_DIR = "elton_john_lyrics"

    if not os.path.exists(LYRICS_DIR) or not os.listdir(LYRICS_DIR):
        return

    # Load lyrics and extract years
    docs, filenames, years = [], [], []
    for filename in os.listdir(LYRICS_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(LYRICS_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
                docs.append(text)
                filenames.append(filename.replace(".txt", ""))

                match = re.search(r"\((\d{4})\)", filename)
                year = int(match.group(1)) if match else None
                years.append(year)

    # Preprocessing with lemmatization and custom stopwords
    stop_words = set(stopwords.words("english"))
    custom_stopwords = set([
        'oh', 'yeah', 'na', 'la', 'hey', 'ooh', 'ah', 'baby', 'gonna',
        'chorus', 'verse', 'repeat', 'refrain'
    ])
    stop_words.update(custom_stopwords)
    lemmatizer = WordNetLemmatizer()

    texts = [
        [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(doc)
         if word.isalpha() and word.lower() not in stop_words]
        for doc in docs
    ]

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=2, no_above=0.8)  
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Find optimal number of topics
    def find_best_lda(corpus, dictionary, texts, start=2, end=15):
        best_model = None
        best_score = -1
        best_k = start
        scores = []
        for k in range(start, end + 1):
            model = models.LdaModel(
                corpus, num_topics=k, id2word=dictionary,
                passes=50, iterations=500, alpha='auto', eta='auto', random_state=42
            )
            coherence = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v').get_coherence()
            scores.append((k, coherence))
            print(f"Coherence for {k} topics: {coherence:.4f}")
            if coherence > best_score:
                best_model = model
                best_score = coherence
                best_k = k

        df_scores = pd.DataFrame(scores, columns=["NumTopics", "Coherence"])
        df_scores.to_csv("lda_coherence_scores.csv", index=False)
        print("Saved coherence scores to lda_coherence_scores.csv")

        plt.figure(figsize=(8, 5))
        plt.plot(df_scores["NumTopics"], df_scores["Coherence"], marker="o")
        plt.title("LDA Coherence Scores by Number of Topics")
        plt.xlabel("Number of Topics")
        plt.ylabel("Coherence Score")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("lda_coherence_plot.png")
        print("Saved plot: lda_coherence_plot.png")

        print(f"Selected {best_k} topics (best coherence: {best_score:.4f})")
        return best_model, best_k

    lda_model, num_topics = find_best_lda(corpus, dictionary, texts)

    
    print("LDA Topics:")
    for idx, topic in lda_model.print_topics(-1):
        print(f"Topic {idx}: {topic}")

    topic_assignments = []
    for i, bow in enumerate(corpus):
        topic_probs = lda_model.get_document_topics(bow)
        dominant_topic = sorted(topic_probs, key=lambda x: x[1], reverse=True)[0][0]
        topic_assignments.append({
            "document": filenames[i],
            "year": years[i],
            "topic": dominant_topic
        })

    df_topics = pd.DataFrame(topic_assignments)
    df_topics.to_csv("lda_topics.csv", index=False)
    print("Saved topic assignments to lda_topics.csv")

    topics = lda_model.show_topics(formatted=False)
    topic_data = [{"Topic": topic_id, "Representation": [word for word, _ in terms]} for topic_id, terms in topics]
    pd.DataFrame(topic_data).to_csv("lda_topic_terms.csv", index=False)

    topic_counts = df_topics["topic"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(topic_counts.index.astype(str), topic_counts.values, color="steelblue")
    plt.xlabel("Topic")
    plt.ylabel("Number of Documents")
    plt.title("Overall LDA Topic Distribution")
    plt.tight_layout()
    plt.savefig("lda_topic_distribution.png")
    print("Saved plot: lda_topic_distribution.png")

    df_years = df_topics.dropna().groupby(["year", "topic"]).size().unstack(fill_value=0)
    df_years.plot(marker='o', figsize=(10, 6))
    plt.title("LDA Topic Trends Over Time")
    plt.xlabel("Year")
    plt.ylabel("Number of Songs")
    plt.legend(title="Topic")
    plt.tight_layout()
    plt.savefig("lda_topic_trend.png")
    print("Saved plot: lda_topic_trend.png")

if __name__ == "__main__":
    main()
