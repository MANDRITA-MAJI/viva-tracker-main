from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from langchain_community.vectorstores import Chroma
import os

path = "data/geo_cleaned.pdf"
db_path = "data/chroma_db"

if not os.path.exists(path):
    print(f"Error: PDF not found")  
else:
    loader = PyPDFLoader(path)
    pages = loader.load()
    full_text = " ".join([page.page_content for page in pages])
    embeddings = HuggingFaceEmbeddings(model_name ="sentence-transformers/all-MiniLM-L6-v2")
    semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type = "percentile",  breakpoint_threshold_amount  = 80)
    docs =  semantic_chunker.create_documents([full_text])

    vectorstore = Chroma.from_documents(documents = docs,
        embedding = embeddings,
        persist_directory = db_path
    )