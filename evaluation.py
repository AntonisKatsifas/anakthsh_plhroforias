import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")

# Load inverted index
def load_inverted_index(filename="inverted_index.json"):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading inverted index: {e}")
        return None

# Load articles
def load_articles(filename="processed_wikipedia_articles.json"):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            articles = json.load(file)
            return articles
    except Exception as e:
        print(f"Error loading articles: {e}")
        return None


# Query processing
def preprocess_query(query):
    query = re.sub(r'[^a-zA-Z\s]', '', query)  
    query = query.lower()  
    tokens = query.split() 
    return tokens

# Boolean search
def boolean_search(query_tokens, inverted_index):
    results = set()
    operation = "OR"  # if there are not logical operators OR is used

    for token in query_tokens:
        if token.upper() in ["AND", "OR", "NOT"]:
            operation = token.upper()
        else:
            postings = set(inverted_index.get(token, []))
            
            if operation == "OR":
                results |= postings
            elif operation == "AND":
                results &= postings
            elif operation == "NOT":
                results -= postings

    return results

#  TF-IDF search
def tfidf_search(query_tokens, articles):

    # Get all document content
    documents = [article['content'] for article in articles]
    
    # Create vectorizer TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    
    # Create query vector
    query = ' '.join(query_tokens)
    query_vector = vectorizer.transform([query])

    # cosine similarity
    cosine_similarities = np.array(X.dot(query_vector.T).toarray()).flatten()
    
    return cosine_similarities

def bm25_search(query_tokens, articles):
    # Tokenize the articles' text
    tokenized_articles = [word_tokenize(article['content'].lower()) for article in articles]
    
    # Initialize the BM25 model
    bm25 = BM25Okapi(tokenized_articles)
    
    # Score the query against all documents
    scores = bm25.get_scores(query_tokens)
    
    return scores

# Evaluation
def evaluate_search_algorithm(query, relevant_docs, inverted_index, articles, algorithm):

    query_tokens = preprocess_query(query)
    
    if algorithm == "boolean":
        retrieved_docs = boolean_search(query_tokens, inverted_index)
    elif algorithm == "tfidf":
        cosine_similarities = tfidf_search(query_tokens, articles)
        ranked_docs = sorted(enumerate(cosine_similarities), key=lambda x: x[1], reverse=True)
        retrieved_docs = [doc_id for doc_id, _ in ranked_docs]
    elif algorithm == "bm25":
        scores = bm25_search(query_tokens, articles)
        ranked_docs = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        retrieved_docs = [doc_id for doc_id, _ in ranked_docs]
    else:
        return None

    # Υπολογισμός ακρίβειας, ανάκλησης και F1-score
    # if there are not related articles 0 the variables
    y_true = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved_docs]
    y_pred = [1 if doc_id in relevant_docs else 0 for doc_id in retrieved_docs]

    if sum(y_true) == 0 and sum(y_pred) == 0:
        precision = recall = f1 = 0.0
    else:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

    # Calculate (MAP)
    if len(retrieved_docs) == 0:
        ap = 0.0
    else:
        ap = average_precision_score(y_true, y_pred)

    return precision, recall, f1, ap

# Create test queries, uncomment one of them only to test cases
def generate_test_queries():
    # test_queries = [
    #     {"query": "machine learning", "relevant_docs": [0, 9, 7]},
    #     {"query": "information retrieval", "relevant_docs": [2, 5, 4]},
    #     {"query": "natural language processing", "relevant_docs": [3, 7, 8]},
    #     {"query": "deep learning", "relevant_docs": [0, 10, 7]},
    #     {"query": "search algorithm", "relevant_docs": [11, 2]},
    #     {"query": "recommender system", "relevant_docs": [6, 4]},
    #     {"query": "prompt engineering", "relevant_docs": [8, 7]},
    #     {"query": "learning to rank", "relevant_docs": [4, 2]},
    # ]
    test_queries = [
        {"query": "What is machine learning?", "relevant_docs": [0, 9]},
        {"query": "Techniques for recommending items", "relevant_docs": [6, 8]},
        {"query": "How to process natural language?", "relevant_docs": [3, 7]},
        {"query": "Algorithms for ranking results", "relevant_docs": [4, 5]},
        {"query": "What are large language models used for?", "relevant_docs": [7, 8]},
        {"query": "Deep learning in AI systems", "relevant_docs": [0, 1]},
        {"query": "How to design search algorithms?", "relevant_docs": [11, 4]},
        {"query": "Applications of recurrent neural networks", "relevant_docs": [9, 10]},
        {"query": "Methods for improving information retrieval", "relevant_docs": [5, 3]},
        {"query": "How to engineer effective prompts?", "relevant_docs": [8, 7]}
    ]
    # test_queries = [
    #     {"query": "applications of deep learning in natural language processing", "relevant_docs": [0, 3, 9]},
    #     {"query": "methods for document classification and information retrieval", "relevant_docs": [1, 2, 4]},
    #     {"query": "techniques in recommender systems and large language models", "relevant_docs": [6, 7, 8]},
    #     {"query": "overview of machine learning and recurrent neural networks", "relevant_docs": [9, 10]},
    #     {"query": "ranking algorithms in search engines", "relevant_docs": [11, 2, 4]},
    #     {"query": "climate change impact on global ecosystems", "relevant_docs": [12]},
    #     {"query": "renaissance art history and its influence", "relevant_docs": [13]},
    #     {"query": "importance of balanced diet in health and nutrition", "relevant_docs": [14]},
    #     {"query": "history and significance of the Olympic Games", "relevant_docs": [15]},
    #     {"query": "exploration techniques in cave environments", "relevant_docs": [16]}
    # ]
    # test_queries = [
    #     {"query": "machine AND learning", "relevant_docs": [0, 10]},
    #     {"query": "deep learning OR artificial intelligence", "relevant_docs": [2, 8, 10]},
    #     {"query": "natural language processing NOT deep learning", "relevant_docs": [3, 4, 5]},
    #     {"query": "machine learning AND (deep learning OR neural networks) NOT artificial intelligence", "relevant_docs": [0, 2, 8]}
    # ]
    # test_queries = [
    #     {"query": "deep learning AND artificial intelligence AND health and nutrition", "relevant_docs": [0, 14, 16]},
    #     {"query": "machine learning AND climate change AND renaissance art", "relevant_docs": [9, 12, 13]},
    #     {"query": "information retrieval AND history of art AND cave exploration", "relevant_docs": [2, 13, 16]},
    #     {"query": "machine learning AND recommender systems AND neural networks", "relevant_docs": [9, 6, 10]},
    #     {"query": "deep learning AND large language models AND health and nutrition", "relevant_docs": [0, 7, 14]},
    #     {"query": "search algorithms AND climate change AND prompt engineering", "relevant_docs": [11, 12, 8]},
    #     {"query": "natural language processing AND document classification AND deep learning AND cave exploration", "relevant_docs": [3, 1, 0, 16]},
    #     {"query": "ranking algorithms AND recommender systems AND health and nutrition", "relevant_docs": [11, 6, 14]},
    #     {"query": "history of art AND large language models AND artificial intelligence", "relevant_docs": [13, 7, 0]},
    #     {"query": "machine learning AND deep learning NOT artificial intelligence", "relevant_docs": [9, 0]},
    #     {"query": "natural language processing AND recommender systems NOT machine learning", "relevant_docs": [3, 6]},
    #     {"query": "climate change AND health and nutrition NOT deep learning", "relevant_docs": [12, 14]},
    #     {"query": "history of art AND health and nutrition AND artificial intelligence", "relevant_docs": [13, 14, 0]},
    #     {"query": "climate change AND cave exploration AND prompt engineering", "relevant_docs": [12, 16, 8]},
    #     {"query": "renaissance art AND machine learning AND recommender systems", "relevant_docs": [13, 9, 6]},
    #     {"query": "recommender systems", "relevant_docs": [6]},
    #     {"query": "ranking algorithms", "relevant_docs": [11]},
    #     {"query": "health", "relevant_docs": [14]},
    #     {"query": "deep learning AND large language models AND healthcare policies", "relevant_docs": [0, 7, 14]},
    #     {"query": "artificial intelligence AND climate change AND medieval history", "relevant_docs": [0, 12, 13]},
    #     {"query": "natural language processing AND cave exploration AND health", "relevant_docs": [3, 16, 14]}
    # ]

    return test_queries

# Evaluate for all algorithms
def evaluate_system():
    inverted_index = load_inverted_index()
    articles = load_articles()

    if not inverted_index or not articles:
        print("Error loading data. Exiting...")
        return

    test_queries = generate_test_queries()

    algorithms = ["boolean", "tfidf", "bm25"]
    for algorithm in algorithms:
        print(f"\nEvaluating using {algorithm} search...")
        
        precision_total = recall_total = f1_total = ap_total = 0
        num_queries = len(test_queries)
        
        for query_data in test_queries:
            query = query_data["query"]
            relevant_docs = query_data["relevant_docs"]
            
            precision, recall, f1, ap = evaluate_search_algorithm(query, relevant_docs, inverted_index, articles, algorithm)
            
            precision_total += precision
            recall_total += recall
            f1_total += f1
            ap_total += ap

            print(f"Query: '{query}' - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}, MAP: {ap:.4f}")
        
        print(f"\nAverage Precision: {precision_total / num_queries:.4f}")
        print(f"Average Recall: {recall_total / num_queries:.4f}")
        print(f"Average F1-Score: {f1_total / num_queries:.4f}")
        print(f"Average MAP: {ap_total / num_queries:.4f}")


def main():
    evaluate_system()

if __name__ == "__main__":
    main()
