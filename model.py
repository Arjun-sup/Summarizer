import torch
import torch.nn as nn
from typing import List, Dict
import os
import requests
from dotenv import load_dotenv
from transformers import AutoTokenizer
from utils import chunk_by_tokens

# Load environment variables
load_dotenv()

class SentenceEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)  # Better representation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        output, (h_n, _) = self.lstm(emb)
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)  # Concatenate directions
        return self.fc(h_n)

def tokenize(sentence: str, vocab: Dict[str, int], max_len: int = 30) -> List[int]:
    """Tokenize a sentence using the vocabulary."""
    tokens = [vocab.get(word.lower(), vocab["<UNK>"]) for word in sentence.split()]
    if not tokens:
        return [vocab["<PAD>"]] * max_len
    return tokens[:max_len] + [vocab["<PAD>"]] * (max_len - len(tokens))

class Summarizer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
        self.api_key = os.getenv("HUGGINGFACE_API_KEY")
        if not self.api_key:
            raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")

    def summarize_offline(self, sentences: List[str], vocab: Dict[str, int],
                          model: SentenceEncoder, compression: str = "medium") -> str:
        """Offline summarization using the LSTM model."""
        max_len = 30
        model.eval()
        try:
            input_tensor = torch.tensor(
                [tokenize(s, vocab, max_len) for s in sentences],
                dtype=torch.long,
                device=self.device
            )
            with torch.no_grad():
                sentence_scores = model(input_tensor).norm(dim=1)
                k = {"high": 0.1, "medium": 0.2, "low": 0.35}[compression]
                num_sentences = max(1, int(len(sentences) * k))
                top_indices = torch.topk(sentence_scores, num_sentences).indices.cpu().numpy()
                return "\n".join([sentences[i] for i in sorted(top_indices)])
        except Exception as e:
            raise Exception(f"Error in offline summarization: {str(e)}")

    def summarize_online(self, text: str, compression: str = "medium") -> str:
        """Online summarization using Hugging Face API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        chunks = chunk_by_tokens(text, self.tokenizer)
        length_map = {"high": 100, "medium": 200, "low": 300}
        max_len = length_map.get(compression, 200)
        summaries = []
        try:
            for chunk in chunks:
                payload = {
                    "inputs": chunk,
                    "parameters": {
                        "do_sample": False,
                        "max_length": max_len,
                        "min_length": max(50, max_len // 2)
                    }
                }
                response = requests.post(
                    "https://api-inference.huggingface.co/models/sshleifer/distilbart-cnn-12-6",
                    headers=headers,
                    json=payload
                )
                if response.status_code == 200:
                    response_data = response.json()
                    if isinstance(response_data, list) and response_data:
                        summary = response_data[0].get('summary_text', '')
                        summaries.append(summary)
                    else:
                        raise ValueError(f"Unexpected API response: {response_data}")
                else:
                    raise Exception(f"API error: {response.status_code} {response.text}")
            return "\n\n".join(summaries)
        except Exception as e:
            raise Exception(f"Error in online summarization: {str(e)}")
