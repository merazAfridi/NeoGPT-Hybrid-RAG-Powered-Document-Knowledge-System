from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

def test_imports():
    assert PyPDFLoader is not None
    assert Docx2txtLoader is not None
    assert CSVLoader is not None
    assert FAISS is not None
    assert OllamaEmbeddings is not None
    assert Ollama is not None

test_imports()