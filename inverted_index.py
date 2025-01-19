import json
from collections import defaultdict

# Loads preprocessed articles from a JSON file
def load_processed_articles(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            articles = json.load(file)
            print(f"Loaded {len(articles)} articles from '{filename}'.")
            return articles
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

# Creates an inverted index from a list of articles
def create_inverted_index(articles):
    inverted_index = defaultdict(list)
    for article_id, article in enumerate(articles):
        for word in article.get('content', '').split():
            if article_id not in inverted_index[word]:
                inverted_index[word].append(article_id)
    return inverted_index

# Saves the inverted index to a JSON file
def save_inverted_index(inverted_index, filename="inverted_index.json"):
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(inverted_index, file, ensure_ascii=False, indent=4)
        print(f"Inverted index saved in '{filename}'.")
    except Exception as e:
        print(f"Error saving file: {e}")


def main():
    input_filename = "processed_wikipedia_articles.json"
    output_filename = "inverted_index.json"
    articles = load_processed_articles(input_filename)
    save_inverted_index(create_inverted_index(articles), output_filename)

if __name__ == "__main__":
    main()
