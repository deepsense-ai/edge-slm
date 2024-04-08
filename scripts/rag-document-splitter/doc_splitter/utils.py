from transformers import AutoTokenizer


class TokensLengthCalculator:
    """A class responsible for calculating the token length of text in string."""
    def __init__(self, embedding_model_name:str) -> None:
        self._tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)


    def __call__(self, text: str) -> int:
        tokens_count = self._tokenizer(text, return_length=True)["length"][0]
        return tokens_count
