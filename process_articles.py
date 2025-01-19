import json
import re

STOP_WORDS = {
    "a", "an", "and", "the", "is", "in", "it", "of", "to", "with", "on", "for", "this", "that", "at", "by", "from",
    "as", "are", "be", "or", "but", "not", "was", "which", "so", "if", "can", "will", "would", "has", "have", "had"
}

def preprocess_text(text):
    """Preprocesses text by removing special characters, converting to lowercase, and remove stop-words."""
    return ' '.join(
        token for token in re.sub(r'[^a-zA-Z\s]', '', text).lower().split() 
        if token not in STOP_WORDS
    )

def read_json_file(filename):
    """Reads JSON file and returns its content or None on failure."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error: {e}")
        return None

def process_articles(input_file, output_file):
    """Processes articles from input JSON file and saves preprocessed data to output JSON file."""
    if articles := read_json_file(input_file):
        processed_articles = [
            {'title': article.get('title', 'No Title'), 
             'content': preprocess_text(article.get('content', ''))}
            for article in articles
        ]
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_articles, f, ensure_ascii=False, indent=4)
        print(f"Processed articles saved in '{output_file}'.")
    else:
        print("No articles to process.")
