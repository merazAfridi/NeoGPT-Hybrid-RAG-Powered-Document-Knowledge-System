## NeoGPT: Hybrid RAG-Powered Document Knowledge System
It combines Hybrid Retrieval-Augmented Generation (RAG) with Knowledge Graphs to deliver deliver accurate, context-aware insights.

#### Project Overview
NeoGPT processes multi-format documents (PDF, DOCX, CSV) to extract, analyze, and visualize knowledge. It integrates semantic search with entity-relationship mapping, enabling advanced question-answering and contextual understanding.
#### How It Works:
###### 1.	Document Processing: 
Upload and process files using pdfplumber, python-docx, and pandas, with RecursiveCharacterTextSplitter for efficient text chunking.
###### 2.	Vector Store Creation: 
Generate embeddings using sentence-transformers/all-MiniLM-L6-v2, stored in FAISS with CUDA acceleration for fast searches.
###### 3.	Knowledge Graphs: 
Extract entities and relationships using SpaCy, visualize with NetworkX, and export graphs.
###### 4.	Hybrid RAG Architecture: 
Combine FAISS-based vector search with knowledge graph traversal, using LLaMA 2 for responses.
###### 5.	Asynchronous Processing: 
Process large datasets efficiently with multi-threaded file handling.


#### Use Cases:

It helps organizations analyze large document sets, enabling: 
•	Contextual question-answering 
•	Knowledge discovery from entity relationships 
•	Semantic search across multi-format datasets

#### Core Technologies:
•	Backend: Python/Flask
•	LLM Integration: LLaMA 2 via Ollama
•	Embedding Model: sentence-transformers/all-MiniLM-L6-v2
•	Vector Store: FAISS with CUDA acceleration
•	NLP Pipeline: SpaCy (en_core_web_sm)
•	Knowledge Graphs: NetworkX
•	Document Processing: pdfplumber, python-docx, pandas

By running all processes locally, NeoGPT ensures user data privacy while delivering efficient and accurate results. With its intuitive interface and robust backend, NeoGPT serves as a powerful tool for those seeking to gain insights and answer questions from their documents effectively.
