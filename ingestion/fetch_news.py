import requests

NEWS_API_KEY = "7b4d84b36ede4a309023652272225b64"

def fetch_news(query="technology"):
    url = f"https://newsapi.org/v2/everything?q={query}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    response = requests.get(url)
    data = response.json()
    articles = []

    for article in data.get("articles", []):
        if article.get("content"):
            articles.append({
                "title": article["title"],
                "content": article["content"],
                "source": article["source"]["name"]
            })
    return articles