import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("lyrics_analysis_combined.csv")

df = df.dropna(subset=[
    "year_x", "avg_sentence_length", "lexical_diversity", "avg_word_length",
    "nouns", "verbs", "adjectives", "named_entities", "modal_verbs",
    "personal_pronouns", "question_sentences", "topic"
])

df["decade"] = (df["year_x"] // 10) * 10

features = [
    "avg_sentence_length", "lexical_diversity", "avg_word_length",
    "nouns", "verbs", "adjectives", "named_entities", "modal_verbs",
    "personal_pronouns", "question_sentences", "topic"
]
X = df[features]
y = df["decade"]

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, multi_class='multinomial')
}

for model_name, model in models.items():
    print(f"\n=== {model_name} ===")
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model)
    ])
    
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    print("Cross-validated accuracy scores:", scores)
    print("Average accuracy:", scores.mean())

    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    
    print("\nClassification Report (cross-validated):")
    print(classification_report(y, y_pred))
    
    cm = confusion_matrix(y, y_pred, labels=np.sort(y.unique()))
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.sort(y.unique()),
                yticklabels=np.sort(y.unique()))
    plt.xlabel("Predicted Decade")
    plt.ylabel("True Decade")
    plt.title(f"Confusion Matrix - {model_name} (Cross-Validated)")
    plt.tight_layout()
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix for {model_name} to {filename}")
