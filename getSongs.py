import lyricsgenius as genius
import os
import re
import json
import time
from datetime import datetime

def clean_lyrics(text):
    lines = text.splitlines()
    clean_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^\d+\s+Contributor', line):
            continue
        if re.search(r'\(repeat.*\)', line, re.IGNORECASE):
            continue
        if re.search(r'\blyrics\b', line, re.IGNORECASE):
            continue
        if re.match(r'^[A-Za-z ]+:$', line):
            continue
        clean_lines.append(line)
    return '\n'.join(clean_lines)

def main():
    with open("token.txt") as f:
        token = f.read().strip()

    api = genius.Genius(token)
    api.timeout = 15
    api.sleep_time = 1
    api.skip_non_songs = True
    api.excluded_terms = ["(Remix)", "(Live)"]
    api.remove_section_headers = True

    artist = api.search_artist("Elton John", max_songs=150, sort="title")
    os.makedirs("elton_john_lyrics", exist_ok=True)

    all_songs = []

    for short_song in artist.songs:
        try:
            full_song = api.song(short_song.id)['song']
            title = short_song.title.replace("/", "_").replace("?", "").replace(":", "").strip()

            release_date = full_song.get('release_date')
            year = None
            if release_date:
                try:
                    year = datetime.strptime(release_date, "%Y-%m-%d").year
                except ValueError:
                    pass

            # Clean and save lyrics
            cleaned_lyrics = clean_lyrics(short_song.lyrics)

            filename = f"{title} ({year})" if year else title
            with open(f"elton_john_lyrics/{filename}.txt", "w", encoding="utf-8") as f:
                f.write(cleaned_lyrics)

            all_songs.append({
                "title": title,
                "year": year,
                "lyrics": cleaned_lyrics
            })

            print(f"Saved: {filename}")
            time.sleep(1)

        except Exception as e:
            print(f"Error fetching full song info for {short_song.title}: {e}")

    with open("elton_john_songs.json", "w", encoding="utf-8") as f:
        json.dump(all_songs, f, ensure_ascii=False, indent=2)

    print("All lyrics and metadata saved.")

if __name__ == "__main__":
    main()
