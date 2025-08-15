import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os
from dotenv import load_dotenv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from utils import extract_pdf_text, split_into_sentences, build_vocab, get_device
from model import SentenceEncoder, Summarizer

# Load environment variables
load_dotenv()

class PDFSummarizerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.summarizer = Summarizer()
        self.device = get_device()
        self.setup_ui()

    def setup_ui(self):
        self.title("Advanced PDF Summarizer")
        self.geometry("600x400")
        self.configure(bg="#f5f5f5")
        main_container = tk.Frame(self, bg="#f5f5f5")
        main_container.pack(padx=20, pady=20, fill="both", expand=True)
        title_label = tk.Label(main_container, text="PDF Summarizer",
                               font=("Helvetica", 24, "bold"),
                               bg="#f5f5f5", fg="#2c3e50")
        title_label.pack(pady=(0, 20))
        self.setup_file_frame(main_container)
        self.setup_options_frame(main_container)
        self.setup_action_buttons(main_container)
        self.setup_progress_section(main_container)

    def setup_file_frame(self, container):
        file_frame = tk.LabelFrame(container, text="File Selection", bg="#f5f5f5", padx=10, pady=5)
        file_frame.pack(fill="x", pady=(0, 10))
        self.file_label = tk.Label(file_frame, text="No file selected",
                                   bg="white", relief="sunken", anchor="w", padx=5)
        self.file_label.pack(side="left", fill="x", expand=True)
        browse_btn = tk.Button(file_frame, text="Browse PDF",
                               command=self.browse_file, bg="#3498db", fg="white", padx=10)
        browse_btn.pack(side="right", padx=(10, 0))

    def setup_options_frame(self, container):
        options_frame = tk.LabelFrame(container, text="Options", bg="#f5f5f5", padx=10, pady=5)
        options_frame.pack(fill="x", pady=(0, 10))
        comp_frame = tk.Frame(options_frame, bg="#f5f5f5")
        comp_frame.pack(fill="x", pady=5)
        tk.Label(comp_frame, text="Compression:", bg="#f5f5f5").pack(side="left")
        self.compression_var = tk.StringVar(value="medium")
        for level in ["high", "medium", "low"]:
            tk.Radiobutton(comp_frame, text=level.capitalize(),
                           variable=self.compression_var, value=level, bg="#f5f5f5").pack(side="left", padx=5)
        mode_frame = tk.Frame(options_frame, bg="#f5f5f5")
        mode_frame.pack(fill="x", pady=5)
        tk.Label(mode_frame, text="Mode:", bg="#f5f5f5").pack(side="left")
        self.mode_var = tk.StringVar(value="online")
        tk.Radiobutton(mode_frame, text="Online (API)",
                       variable=self.mode_var, value="online", bg="#f5f5f5").pack(side="left", padx=5)
        tk.Radiobutton(mode_frame, text="Offline (Local)",
                       variable=self.mode_var, value="offline", bg="#f5f5f5").pack(side="left", padx=5)

    def setup_action_buttons(self, container):
        self.summarize_btn = tk.Button(container, text="Summarize PDF",
                                       command=self.run_summarization,
                                       font=("Helvetica", 12, "bold"),
                                       bg="#2ecc71", fg="white", padx=20, pady=10)
        self.summarize_btn.pack(pady=10)

    def setup_progress_section(self, container):
        self.progress = ttk.Progressbar(container, mode="indeterminate")
        self.progress.pack(fill="x", pady=(5, 0))
        self.status_label = tk.Label(container, text="Ready", font=("Arial", 10),
                                     bg="#f5f5f5", fg="#2c3e50")
        self.status_label.pack(pady=10)

    def browse_file(self):
        file = filedialog.askopenfilename(title="Select PDF File", filetypes=[("PDF Files", "*.pdf")])
        if file:
            self.file_path = file
            self.file_label.config(text=os.path.basename(file))
            self.status_label.config(text="File loaded successfully")

    def generate_pdf(self, summary: str, filename: str = "summary_output.pdf") -> str:
        try:
            doc = SimpleDocTemplate(filename, pagesize=A4)
            styles = getSampleStyleSheet()
            story = []
            title_style = styles["Heading1"]
            story.append(Paragraph("Document Summary", title_style))
            story.append(Spacer(1, 20))
            for para in summary.split("\n"):
                if para.strip():
                    story.append(Paragraph(para.strip(), styles["Normal"]))
                    story.append(Spacer(1, 12))
            doc.build(story)
            return filename
        except Exception as e:
            raise Exception(f"Error generating PDF: {str(e)}")

    def run_summarization(self):
        if not hasattr(self, 'file_path'):
            messagebox.showerror("Error", "Please select a PDF file first.")
            return
        self.summarize_btn.config(state="disabled")
        self.progress.start(10)
        self.status_label.config(text="Processing... Please wait.")
        threading.Thread(target=self._summarize_thread, daemon=True).start()

    def _summarize_thread(self):
        try:
            self.status_label.config(text="Extracting text from PDF...")
            text = extract_pdf_text(self.file_path)
            compression = self.compression_var.get()
            mode = self.mode_var.get()
            if mode == "online":
                self.status_label.config(text="Generating summary using API...")
                summary = self.summarizer.summarize_online(text, compression)
            else:
                self.status_label.config(text="Generating summary offline...")
                sentences = split_into_sentences(text)
                vocab = build_vocab(sentences)
                model = SentenceEncoder(len(vocab), embed_dim=128, hidden_dim=128).to(self.device)
                summary = self.summarizer.summarize_offline(sentences, vocab, model, compression)
            self.status_label.config(text="Generating PDF...")
            output_pdf = self.generate_pdf(summary)
            self.status_label.config(text=f"Summary saved: {output_pdf}")
            messagebox.showinfo("Success", "Summary generated successfully!")
            os.startfile(output_pdf)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
            self.status_label.config(text="Error occurred during summarization.")
        finally:
            self.progress.stop()
            self.summarize_btn.config(state="normal")

if __name__ == "__main__":
    app = PDFSummarizerApp()
    app.mainloop()
