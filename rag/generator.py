from transformers import pipeline

class NewsGenerator:
    def __init__(self):
        self.summarizer = pipeline("text-generation", model="gpt2")

    def generate_summary(self, articles):
        if not articles:
            return "No articles found."
        
        # Combine article texts
        combined_text = " ".join([a.get("content", "") for a in articles])[:512]
        
        # Generate summary
        result = self.summarizer(combined_text, max_length=150, do_sample=True, temperature=0.7)
        return result[0]["generated_text"]