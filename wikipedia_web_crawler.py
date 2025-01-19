import requests
from bs4 import BeautifulSoup
import json


# Search from wikipedia
def search_wikipedia(query):
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'format': 'json',
        'srlimit': 1
    }
    response = requests.get("https://en.wikipedia.org/w/api.php", params=params)
    return response.json().get('query', {}).get('search', []) if response.status_code == 200 else []

# Fetch article's content
def fetch_article_content(title):
    url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
    soup = BeautifulSoup(requests.get(url).content, 'html.parser')
    content = soup.find('div', class_='mw-parser-output')
    return ' '.join([p.text for p in content.find_all('p') if p.text.strip()]) if content else None

# Main function to fetch articles and save them to a json
def fetch_and_save_articles(queries, filename="wikipedia_articles.json"):
    articles = []
    for query in queries:
        search_results = search_wikipedia(query)
        for result in search_results:
            content = fetch_article_content(result['title'])
            print(result['title'])
            if content:
                articles.append({'title': result['title'], 'content': content})
    if articles:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=4)
        print(f"Saved {len(articles)} articles to {filename}.")
    else:
        print("No articles found.")


def main():
    queries = "Deep learning,Document classification,Information retrieval,Natural language processing,Learning to rank,Information retrieval,Recommender system,Large language model,Prompt engineering,Machine learning,Recurrent neural network,Ranking Algorithms,Climate Change,History of Art,Health and Nutrition,The History of the Olympic Games,Cave exploration".split(",")
    fetch_and_save_articles(queries)

if __name__ == "__main__":
    main()
