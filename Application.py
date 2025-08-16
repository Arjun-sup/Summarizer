import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import networkx as nx
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import re
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load environment variables
load_dotenv()

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def preprocess_text(text):
    """Enhanced text preprocessing"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text)
    # Remove page numbers, headers, footers
    text = re.sub(r'\b\d+\b', '', text)
    # Remove URLs and email addresses
    text = re.sub(r'http\S+|www\S+|[\w\.-]+@[\w\.-]+', '', text)
    return text.strip()

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash')

# Text extraction using PyPDF2 (keeping our existing logic)
def extract_pdf_content(file_path):
    try:
        pdf_doc = PdfReader(file_path)
        content_stream = ""
        for page_num in range(len(pdf_doc.pages)):
            page = pdf_doc.pages[page_num]
            content_stream += page.extract_text()
        return content_stream
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Online summarization using Gemini (our existing logic)
def draft_final_prompt(word_size, content, compression):
    compression_map = {
        "high": int(word_size * 0.5),    # 50% of original
        "medium": int(word_size * 0.7),  # 70% of original  
        "low": word_size                 # 100% of original
    }
    target_words = compression_map.get(compression, word_size)
    
    return f"""

<Document Content>
{content}
<Document Content>
"""

def model_answer(ques):
    try:
        config = genai.GenerationConfig(max_output_tokens=2048, temperature=0.5)
        response = model.generate_content(ques, generation_config=config)
        return response.text
    except Exception as e:
        return f"An error occurred: {e}"

# Offline summarization using our existing TextRank logic
def textrank_summarize(text, compression="medium"):
    try:
        sentences = sent_tokenize(text)
        
        if len(sentences) < 3:
            return "Document too short for meaningful summarization."
        
        # Compression level mapping
        compression_map = {
            "high": 0.15,    # 15% of sentences
            "medium": 0.25,  # 25% of sentences
            "low": 0.4       # 40% of sentences
        }
        
        sentence_ratio = compression_map.get(compression, 0.25)
        num_sentences = max(3, min(len(sentences), int(len(sentences) * sentence_ratio)))
        
        # TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Similarity Matrix
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        
        # TextRank
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph, max_iter=100, tol=1e-4)
        
        # Rank and select sentences
        ranked_sentences = sorted(((scores[i], i, s) for i, s in enumerate(sentences)), reverse=True)
        selected_indices = sorted([idx for _, idx, _ in ranked_sentences[:num_sentences]])
        summary_sentences = [sentences[i] for i in selected_indices]
        
        return " ".join(summary_sentences)
        
    except Exception as e:
        return f"Offline summarization failed: {e}"

# Enhanced offline summarization
def enhanced_textrank_summarize(text, compression="medium"):
    try:
        # Preprocess text
        text = preprocess_text(text)
        sentences = sent_tokenize(text)
        
        if len(sentences) < 3:
            return "Document too short for meaningful summarization."
        
        # Filter out very short sentences
        sentences = [s for s in sentences if len(s.split()) > 5]
        
        compression_map = {
            "high": 0.15, "medium": 0.25, "low": 0.4
        }
        
        sentence_ratio = compression_map.get(compression, 0.25)
        num_sentences = max(3, min(len(sentences), int(len(sentences) * sentence_ratio)))
        
        # Enhanced TF-IDF with better parameters
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=2000,
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,
            max_df=0.8
        )
        tfidf_matrix = vectorizer.fit_transform(sentences)
        
        # Cosine similarity instead of dot product
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Apply threshold to reduce noise
        threshold = 0.1
        similarity_matrix[similarity_matrix < threshold] = 0
        
        # TextRank with damping factor
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph, alpha=0.85, max_iter=100, tol=1e-4)
        
        # Position bias - prefer sentences from beginning and end
        position_weights = []
        for i in range(len(sentences)):
            if i < len(sentences) * 0.2:  # First 20%
                weight = 1.2
            elif i > len(sentences) * 0.8:  # Last 20%
                weight = 1.1
            else:
                weight = 1.0
            position_weights.append(weight)
        
        # Combine TextRank scores with position weights
        final_scores = {i: scores[i] * position_weights[i] for i in range(len(sentences))}
        
        # Select diverse sentences (avoid redundancy)
        selected_sentences = []
        ranked_sentences = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        for idx, score in ranked_sentences:
            if len(selected_sentences) >= num_sentences:
                break
            
            # Check similarity with already selected sentences
            is_diverse = True
            current_sentence = sentences[idx]
            
            for selected_idx in selected_sentences:
                selected_sentence = sentences[selected_idx]
                # Simple word overlap check
                words1 = set(current_sentence.lower().split())
                words2 = set(selected_sentence.lower().split())
                overlap = len(words1.intersection(words2)) / len(words1.union(words2))
                
                if overlap > 0.5:  # Too similar
                    is_diverse = False
                    break
            
            if is_diverse:
                selected_sentences.append(idx)
        
        # Sort by original order and create summary
        selected_sentences.sort()
        summary_sentences = [sentences[i] for i in selected_sentences]
        
        return " ".join(summary_sentences)
        
    except Exception as e:
        return f"Enhanced offline summarization failed: {e}"

def hybrid_summarize(text, compression="medium"):
    """Combine multiple summarization techniques"""
    try:
        # Get summaries from different methods
        textrank_summary = enhanced_textrank_summarize(text, compression)
        frequency_summary = frequency_based_summarize(text, compression)
        
        # Combine and rank sentences from both methods
        all_sentences = set()
        
        if "failed" not in textrank_summary.lower():
            all_sentences.update(sent_tokenize(textrank_summary))
        
        if "failed" not in frequency_summary.lower():
            all_sentences.update(sent_tokenize(frequency_summary))
        
        if not all_sentences:
            return textrank_summary  # Fallback
        
        # Re-rank combined sentences
        combined_text = " ".join(all_sentences)
        return combined_text
        
    except Exception as e:
        return enhanced_textrank_summarize(text, compression)

def frequency_based_summarize(text, compression="medium"):
    """Frequency-based summarization as complement"""
    try:
        sentences = sent_tokenize(text)
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Calculate word frequencies
        stop_words = set(stopwords.words('english'))
        word_freq = Counter([w for w in words if w not in stop_words and len(w) > 2])
        
        # Score sentences
        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words_in_sentence = re.findall(r'\b\w+\b', sentence.lower())
            score = sum(word_freq.get(word, 0) for word in words_in_sentence if word not in stop_words)
            sentence_scores[i] = score / len(words_in_sentence) if words_in_sentence else 0
        
        # Select top sentences
        compression_map = {"high": 0.15, "medium": 0.25, "low": 0.4}
        num_sentences = max(3, int(len(sentences) * compression_map.get(compression, 0.25)))
        
        top_indices = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:num_sentences]
        top_indices = sorted([idx for idx, score in top_indices])
        
        return " ".join([sentences[i] for i in top_indices])
        
    except Exception as e:
        return f"Frequency-based summarization failed: {e}"

# PDF generation
def generate_pdf(summary, filename="summary_output.pdf"):
    try:
        doc = SimpleDocTemplate(filename, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []
        
        # Add title
        title = Paragraph("PDF Summary", styles["Title"])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Add summary content
        paragraphs = summary.split('\n')
        for para in paragraphs:
            if para.strip():
                p = Paragraph(para.strip(), styles["Normal"])
                story.append(p)
                story.append(Spacer(1, 12))
        
        doc.build(story)
        return filename
    except Exception as e:
        raise Exception(f"PDF generation failed: {e}")

# Desktop Application Class
class PDFSummarizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("PDF Summarizer - Desktop App")
        self.geometry("500x400")
        self.resizable(False, False)
        self.configure(bg="#f0f0f0")
        
        self.file_path = None
        self.setup_ui()
    
    def setup_ui(self):
        # Title
        title_label = tk.Label(self, text="PDF Summarizer", 
                              font=("Helvetica", 18, "bold"), 
                              bg="#f0f0f0", fg="#2c3e50")
        title_label.pack(pady=(20, 15))
        
        # File selection frame
        file_frame = tk.Frame(self, bg="#f0f0f0")
        file_frame.pack(pady=(0, 15), fill="x", padx=20)
        
        self.file_label = tk.Label(file_frame, text="No file selected", 
                                  bg="#ffffff", relief="sunken", 
                                  anchor="w", width=45, height=2)
        self.file_label.pack(side="left", fill="x", expand=True)
        
        browse_btn = tk.Button(file_frame, text="Browse PDF", 
                              command=self.browse_file, 
                              bg="#3498db", fg="white", 
                              font=("Arial", 10, "bold"),
                              width=12, height=2)
        browse_btn.pack(side="right", padx=(10, 0))
        
        # Compression level frame
        comp_frame = tk.Frame(self, bg="#f0f0f0")
        comp_frame.pack(pady=(0, 15))
        
        tk.Label(comp_frame, text="Compression Level:", 
                font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=(0, 5))
        
        compression_frame = tk.Frame(comp_frame, bg="#f0f0f0")
        compression_frame.pack()
        
        self.compression_var = tk.StringVar(value="medium")
        for level in ["high", "medium", "low"]:
            tk.Radiobutton(compression_frame, text=level.capitalize(), 
                          variable=self.compression_var, value=level,
                          bg="#f0f0f0", font=("Arial", 10)).pack(side="left", padx=10)
        
        # Mode selection frame
        mode_frame = tk.Frame(self, bg="#f0f0f0")
        mode_frame.pack(pady=(0, 15))
        
        tk.Label(mode_frame, text="Processing Mode:", 
                font=("Arial", 12, "bold"), bg="#f0f0f0").pack(pady=(0, 5))
        
        mode_buttons_frame = tk.Frame(mode_frame, bg="#f0f0f0")
        mode_buttons_frame.pack()
        
        self.mode_var = tk.StringVar(value="offline")
        tk.Radiobutton(mode_buttons_frame, text="Offline (TextRank)", 
                      variable=self.mode_var, value="offline", 
                      bg="#f0f0f0", font=("Arial", 10)).pack(side="left", padx=10)
        tk.Radiobutton(mode_buttons_frame, text="Online (Gemini AI)", 
                      variable=self.mode_var, value="online", 
                      bg="#f0f0f0", font=("Arial", 10)).pack(side="left", padx=10)
        
        # Summarize button
        self.summarize_btn = tk.Button(self, text="Generate Summary", 
                                      font=("Arial", 14, "bold"), 
                                      bg="#27ae60", fg="white",
                                      activebackground="#229954", 
                                      cursor="hand2", 
                                      command=self.run_summarization,
                                      width=20, height=2)
        self.summarize_btn.pack(pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(self, mode="indeterminate")
        self.progress.pack(fill="x", padx=50, pady=(10, 0))
        
        # Status label
        self.status_label = tk.Label(self, text="Ready to process PDF files", 
                                    font=("Arial", 10), bg="#f0f0f0", 
                                    fg="#7f8c8d")
        self.status_label.pack(pady=15)
    
    def browse_file(self):
        file = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf")]
        )
        if file:
            self.file_path = file
            filename = os.path.basename(file)
            self.file_label.config(text=f"Selected: {filename}")
            self.status_label.config(text="PDF file selected. Ready to summarize.")
    
    def run_summarization(self):
        if not self.file_path:
            messagebox.showerror("Error", "Please select a PDF file first.")
            return
        
        mode = self.mode_var.get()
        if mode == "online" and not GOOGLE_API_KEY:
            messagebox.showerror("Error", "Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
            return
        
        self.summarize_btn.config(state="disabled")
        self.progress.start(10)
        self.status_label.config(text="Processing PDF... Please wait.")
        
        # Run summarization in separate thread
        threading.Thread(target=self.summarize_in_thread, daemon=True).start()
    
    def summarize_in_thread(self):
        try:
            compression = self.compression_var.get()
            mode = self.mode_var.get()
            
            # Extract text from PDF
            self.update_status("Extracting text from PDF...")
            text = extract_pdf_content(self.file_path)
            
            if not text:
                raise Exception("Failed to extract text from PDF")
            
            # Generate summary
            if mode == "online":
                self.update_status("Generating summary using Gemini AI...")
                # Calculate target word count based on text length
                word_count = len(text.split())
                target_words = min(500, max(100, word_count // 10))
                
                final_prompt = draft_final_prompt(target_words, text, compression)
                summary = model_answer(final_prompt)
            else:
                self.update_status("Generating summary using enhanced algorithms...")
                summary = hybrid_summarize(text, compression)
                
                # Fallback chain
                if "failed" in summary.lower():
                    summary = enhanced_textrank_summarize(text, compression)
                    if "failed" in summary.lower():
                        summary = textrank_summarize(text, compression)
            
            # Generate PDF
            self.update_status("Creating PDF summary...")
            output_filename = f"summary_{compression}_{mode}.pdf"
            output_path = generate_pdf(summary, output_filename)
            
            # Open the generated PDF
            self.update_status(f"Summary saved as: {output_filename}")
            os.startfile(output_path) 
            
            messagebox.showinfo("Success", f"Summary generated successfully!\nFile: {output_filename}")
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.update_status("Error occurred during processing.")
        finally:
            self.progress.stop()
            self.summarize_btn.config(state="normal")
    
    def update_status(self, message):
        self.status_label.config(text=message)
        self.update_idletasks()

if __name__ == "__main__":
    app = PDFSummarizerApp()
    app.mainloop()
