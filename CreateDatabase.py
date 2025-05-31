import os
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def extract_text_from_pdf(pdf_path):
    try:
        # Ensure it's actually a PDF
        with open(pdf_path, "rb") as f:
            header = f.read(5)  # Read first few bytes
            if not header.startswith(b"%PDF-"):
                print(f"Skipping non-PDF file: {pdf_path}")
                return ""  # Ignore non-PDFs
        
        doc = fitz.open(pdf_path)
        # Check if the PDF is encrypted
        if doc.is_encrypted:
            print(f"Skipping encrypted file: {pdf_path}")
            return ""  # Ignore encrypted PDFs

        text = "\n".join([page.get_text() for page in doc.pages()])
        
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""  # Return empty string instead of crashing

def get_all_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = []

for pdf_path in get_all_files("D:/My Vault/Scholarly  Resources/Viking History Books"):
    text = extract_text_from_pdf(pdf_path)
    if text.strip():
        raw_doc = Document(
            page_content=text,
            metadata={"source": os.path.basename(pdf_path)}
        )
        doc_chunks = splitter.split_documents([raw_doc])
        documents.extend(doc_chunks)
texts = [doc.page_content for doc in documents]  # Extracting only the text content
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = FAISS.from_texts(texts, embedding_model)
db.save_local("pdf_vector_store")
print("Done")