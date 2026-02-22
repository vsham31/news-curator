# AI Curated News RAG System

## Setup

1. Create virtual environment
2. Install dependencies:
   pip install -r requirements.txt

3. Add your NewsAPI key in ingestion/fetch_news.py

4. Run the server:
   uvicorn api.main:app --reload

5. Open:
   http://127.0.0.1:8000/curated-news?query=ai