from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class NewsGenerator:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate_summary(self, articles):
        if not articles:
            return "No articles found."

        combined_text = " ".join([a.get("content", "") for a in articles]).strip()
        if not combined_text:
            return "No article content available to summarize."

        inputs = self.tokenizer(
            combined_text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )

        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_length=150,
            min_length=30,
            num_beams=4,
            early_stopping=True,
        )
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
