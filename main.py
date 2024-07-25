import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
if device == "cuda":
    print(torch.cuda.get_device_name(0))

# Define paths and token
origin_model_path = "mistralai/Mistral-7B-Instruct-v0.1"
model_path = "filipealmeida/Mistral-7B-Instruct-v0.1-sharded"
token = "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"  # Replace with your actual token

# Configuration for 4-bit quantization (GPU only)
if device == "cuda":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config, device_map="auto", token=token)
else:
    model = AutoModelForCausalLM.from_pretrained(model_path, token=token)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(origin_model_path, token=token)

# Setup text generation pipeline
text_generation_pipeline = transformers.pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=300,
    temperature=0.3,
    do_sample=True,
)

mistral_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Load the CSV file
loader = CSVLoader(file_path='/content/drive/MyDrive/customers-100.csv')
data = loader.load()

# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunked_docs = text_splitter.split_documents(data)

# Using HuggingFace embeddings
embeddings = HuggingFaceEmbeddings()

# Create FAISS vector store
db = FAISS.from_documents(chunked_docs, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# Now you can use mistral_llm and retriever for your downstream tasks
