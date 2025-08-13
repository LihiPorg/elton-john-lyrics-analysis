import os
import pandas as pd
from trankit import Pipeline
import re

def extract_year(filename):
    match = re.search(r'\((\d{4})\)', filename)
    return int(match.group(1)) if match else None

def lexical_diversity(words):
    return len(set(words)) / len(words) if words else 0

def main():
    p = Pipeline(lang='english')
    LYRICS_DIR = "elton_john_lyrics"

    data = []

    for filename in os.listdir(LYRICS_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(LYRICS_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()

            result = p(text)
            ner_result = p.ner(text)

            words = []
            num_nouns = num_verbs = num_adjectives = 0
            num_questions = num_pronouns = num_modals = total_word_length = 0
            modal_set = {'can', 'could', 'may', 'might', 'must', 'shall', 'should', 'will', 'would'}

            for sentence in result['sentences']:
                if sentence['text'].strip().endswith('?'):
                    num_questions += 1

                for token in sentence['tokens']:
                    pos = token.get('upos')
                    token_text = token.get('text').lower()
                    words.append(token_text)
                    total_word_length += len(token_text)

                    if pos == 'NOUN':
                        num_nouns += 1
                    elif pos == 'VERB':
                        num_verbs += 1
                    elif pos == 'ADJ':
                        num_adjectives += 1
                    elif pos == 'PRON':
                        num_pronouns += 1
                    if token_text in modal_set:
                        num_modals += 1

            num_entities = sum(len(s.get('entities', [])) for s in ner_result.get('sentences', []))

            year = extract_year(filename)
            num_sentences = len(result['sentences'])
            num_tokens = len(words)

            data.append({
                "song": filename.replace(".txt", ""),
                "year": year,
                "num_tokens": num_tokens,
                "num_sentences": num_sentences,
                "avg_sentence_length": num_tokens / num_sentences if num_sentences else 0,
                "lexical_diversity": lexical_diversity(words),
                "avg_word_length": total_word_length / num_tokens if num_tokens else 0,
                "nouns": num_nouns,
                "verbs": num_verbs,
                "adjectives": num_adjectives,
                "named_entities": num_entities,
                "modal_verbs": num_modals,
                "personal_pronouns": num_pronouns,
                "question_sentences": num_questions
            })

    df = pd.DataFrame(data)
    df.to_csv("lyrics_stylometry_trankit_extended.csv", index=False)
    print(df.head())

if __name__ == "__main__":
    main()
