from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

DEFAULT_MODEL = "facebook/bart-large-cnn"

class Summarizer:
    """
    Summarizer class for abstractive summarization using transformer models.
    """

    def __init__(self, model_name=DEFAULT_MODEL):
        """
        Initialize tokenizer and model.

        Args:
            model_name (str): Hugging Face model name (e.g., BART, PEGASUS).
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text, max_len=200, min_len=50):
        """
        Summarizes the given text.

        Args:
            text (str): Input text to summarize.
            max_len (int): Maximum length of summary.
            min_len (int): Minimum length of summary.
        Returns:
            str: Generated summary.
        """
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)

        # Generate summary using beam search
        summary_ids = self.model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=min_len,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True
        )

        # Decode generated ids back to text
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
