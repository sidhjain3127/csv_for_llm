from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
import pickle
import os

# Use raw string or double backslashes for the file path
file_path = r'C:\Users\laptop\Documents\csv_for_llm\customers-100.csv'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file at {file_path} does not exist.")

# Load the CSV file
loader = CSVLoader(file_path=file_path)
data = loader.load()

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunked_docs = text_splitter.split_documents(data)

# Ensure the output directory exists
output_dir = r'C:\Users\laptop\Documents\csv_for_llm\data'
os.makedirs(output_dir, exist_ok=True)

# Save the chunked documents to disk
output_file = os.path.join(output_dir, 'chunked_docs.pkl')
with open(output_file, 'wb') as f:
    pickle.dump(chunked_docs, f)

print("Documents loaded and split successfully.")
