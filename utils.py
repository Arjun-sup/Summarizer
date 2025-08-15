import re
import torch
from PyPDF2 import PdfReader

def chunk_by_tokens(text, tokenizer, max_tokens=1024):
    """Split text into chunks that fit within max_tokens for a tokenizer."""
    tokens = tokenizer.tokenize(text)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunks.append(tokenizer.convert_tokens_to_string(chunk_tokens))
    return chunks

def extract_pdf_text(pdf_path):
    """Extract text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()

def split_into_sentences(text):
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def build_vocab(sentences):
    """Build a simple word-to-index vocabulary."""
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for sentence in sentences:
        for word in sentence.split():
            word = word.lower()
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

def get_device():
    """Return torch device (cuda or cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")