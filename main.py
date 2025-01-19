from wikipedia_web_crawler import fetch_and_save_articles
from process_articles import process_articles
from inverted_index import load_processed_articles, create_inverted_index, save_inverted_index
from search_engine import search_interface
from evaluation import evaluate_system

def main():
    # Step 1ο
    queries = "Deep learning,Document classification,Information retrieval,Natural language processing,Learning to rank,Information retrieval,Recommender system,Large language model,Prompt engineering,Machine learning,Recurrent neural network,Ranking Algorithms,Climate Change,History of Art,Health and Nutrition,The History of the Olympic Games,Cave exploration".split(",")
    fetch_and_save_articles(queries)
    
    # Step 2ο
    process_articles("wikipedia_articles.json", "processed_wikipedia_articles.json")

    # Step 3ο
    input_filename = "processed_wikipedia_articles.json"
    output_filename = "inverted_index.json"
    articles = load_processed_articles(input_filename)
    save_inverted_index(create_inverted_index(articles), output_filename)

    # Step 4ο
    search_interface()

    # Step 5ο
    evaluate_system()

if __name__ == "__main__":
    main()
