import json
import re
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

def load_json(filename):
    """Load a JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading '{filename}': {e}")
        return None

def preprocess_query(query):
    """Clean and tokenize the query."""
    return re.sub(r'[^a-zA-Z\s]', '', query).lower().split()

def boolean_search(query_tokens, inverted_index):
    """Perform boolean search on the inverted index."""
    results, operation = set(), "OR"
    found_results = False  # Flag to check if we found any results

    for token in query_tokens:
        if token.upper() in ["AND", "OR", "NOT"]:
            operation = token.upper()
        else:
            postings = set(inverted_index.get(token, []))
            if not postings:  # If no postings found for this token, skip it
                continue

            if operation == "OR":
                results |= postings
                found_results = True
            elif operation == "AND":
                if not results:  # If no previous results, initialize with the first set
                    results = postings
                else:
                    results &= postings
            elif operation == "NOT":
                results -= postings

    # If no results are found, return an empty set
    return results if found_results else set()


def tfidf_search(query_tokens, articles, threshold=0.1):
    """Perform TF-IDF search on the articles."""
    corpus = [article['content'] for article in articles]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    query_vector = vectorizer.transform([' '.join(query_tokens)])
    scores = np.dot(query_vector, tfidf_matrix.T).toarray().flatten()

    # Only return results with scores above the threshold
    return [i for i, score in enumerate(scores) if score > threshold]

def bm25_search(query_tokens, articles, threshold=0.1):
    """Perform BM25 search on the articles."""
    corpus = [article['content'].split() for article in articles]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)

    # Only return results with scores above the threshold
    return [i for i, score in enumerate(scores) if score > threshold]


def search_query(query, inverted_index, articles, algorithm):
    """Search the query using the selected algorithm."""
    query_tokens = preprocess_query(query)
    if algorithm == "boolean":
        matched_ids = boolean_search(query_tokens, inverted_index)
        return [articles[doc_id] for doc_id in matched_ids] if matched_ids else []
    elif algorithm == "tfidf":
        matched_ids = tfidf_search(query_tokens, articles)
        return [articles[doc_id] for doc_id in matched_ids] if matched_ids else []
    elif algorithm == "bm25":
        matched_ids = bm25_search(query_tokens, articles)
        return [articles[doc_id] for doc_id in matched_ids] if matched_ids else []
    return []

def search_interface():
    """Search interface to interact with the user."""
    inverted_index = load_json("inverted_index.json")
    articles = load_json("processed_wikipedia_articles.json")
    if not inverted_index or not articles:
        print("Error loading data. Exiting...")
        return

    algorithms = {"1": "boolean", "2": "tfidf", "3": "bm25"}
    while True:
        choice = input("\nChoose algorithm (1: Boolean, 2: TF-IDF, 3: BM25, 'exit' to quit): ")
        if choice.lower() == "exit":
            print("Exiting search engine. Goodbye!")
            break
        algorithm = algorithms.get(choice)
        if not algorithm:
            print("Invalid choice. Try again.")
            continue

        query = input("Enter your query: ")
        results = search_query(query, inverted_index, articles, algorithm)
        if results:
            print(f"\nFound {len(results)} result(s):")
            for result in results:
                print(f"- {result['title']}")
        else:
            print("No results found.")


def main():
    search_interface()

if __name__ == "__main__":
    main()
