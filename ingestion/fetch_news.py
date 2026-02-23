import requests

NEWS_API_KEY = "7b4d84b36ede4a309023652272225b64"


def fetch_news(query="technology"):
    url = (
        "https://newsapi.org/v2/everything"
        f"?q={query}&language=en&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    )

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return {
            "ok": False,
            "error": {
                "type": "newsapi_request_failed",
                "status": status,
                "detail": str(exc)
            }
        }

    articles = []

    for article in data.get("articles", []):
        if article.get("content"):
            articles.append({
                "title": article["title"],
                "content": article["content"],
                "source": article["source"]["name"]
            })
    return {"ok": True, "articles": articles}