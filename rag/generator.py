from transformers import pipeline, GPT2Tokenizer

class NewsGenerator:
    def __init__(self):
        # Upgraded to BART for actual summarization (better than GPT-2 text generation)
        self.summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def generate_summary(self, articles):
        if not articles:
            return "No articles found."
        
        # Combine article texts
        combined_text = " ".join([a.get("content", "") for a in articles])
        
        # Limit to 1024 tokens using actual token counting (better than character limit)
        tokens = self.tokenizer.encode(combined_text)[:1024]
        combined_text = self.tokenizer.decode(tokens)
        
        # Generate summary using BART (trained summarization model)
        result = self.summarizer(combined_text, max_length=150, min_length=30, do_sample=False)
        return result[0]["summary_text"]