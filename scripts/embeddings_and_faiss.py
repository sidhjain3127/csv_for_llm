from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import pickle

# Load the chunked documents
with open(r'C:\Users\laptop\Documents\csv_for_llm\data\chunked_docs.pkl', 'rb') as f:
    chunked_docs = pickle.load(f)

# Using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

# Create FAISS vector store
db = FAISS.from_documents(chunked_docs, embeddings)

# Save the FAISS index
db.save_local('csv_for_llm/data/faiss_index')

print("Embeddings and FAISS index created successfully.")
