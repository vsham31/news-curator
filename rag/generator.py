from transformers import pipeline

class NewsGenerator:
    def __init__(self):
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def generate_summary(self, articles):
        combined_text = " ".join([a["content"] for a in articles])
        summary = self.summarizer(combined_text[:3000], max_length=300, min_length=100, do_sample=False)
        return summary[0]["summary_text"]