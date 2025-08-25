from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredHTMLLoader, CSVLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from typing import List
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

# Load env
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# Splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)

# ✅ Gemini embeddings
embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Chroma vector store
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    elif file_path.endswith('.html'):
        loader = UnstructuredHTMLLoader(file_path)
    elif file_path.endswith('.csv'):
        loader = CSVLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    documents = loader.load()
    return text_splitter.split_documents(documents)

def index_document_to_chroma(file_path_or_content: str, file_id: int) -> bool:
    try:
        if os.path.exists(file_path_or_content):
            splits = load_and_split_document(file_path_or_content)
        else:
            loader = TextLoader(file_path_or_content)
            documents = loader.load()
            splits = text_splitter.split_documents(documents)

        for split in splits:
            split.metadata['file_id'] = file_id

        vectorstore.add_documents(splits)
        return True
    except Exception as e:
        print(f"Error indexing document: {e}")
        return False

def delete_doc_from_chroma(file_id: int):
    try:
        docs = vectorstore.get(where={"file_id": file_id})
        print(f"Found {len(docs['ids'])} chunks for file_id {file_id}")
        vectorstore._collection.delete(where={"file_id": file_id})
        return True
    except Exception as e:
        print(f"Error deleting document: {e}")
        return False
