from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from typing import Optional, Dict, List, Union, Any, Tuple
from datetime import datetime
import logging
import os
import pdfplumber
import docx
import pandas as pd
import torch
import spacy
import networkx as nx
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from entity_extraction import extract_entities, extract_relationships, save_to_csv
from knowledge_graph import KnowledgeGraph
from threading import Lock


load_dotenv() #environment variables

#loadING spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

#logging configure
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


app = Flask(__name__) #initialize Flask app
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  #file size 16MB max 

class ThreadSafeDict:
    def __init__(self):
        self._dict = {}
        self._lock = Lock()

    def __setitem__(self, key, value):
        with self._lock:
            self._dict[key] = value

    def __getitem__(self, key):
        with self._lock:
            return self._dict[key]

    def __contains__(self, key):
        with self._lock:
            return key in self._dict

    def keys(self):
        with self._lock:
            return list(self._dict.keys())

    def get(self, key, default=None):
        with self._lock:
            return self._dict.get(key, default)

    def clear(self):
        with self._lock:
            self._dict.clear()

#thread-safe storage initialized
documents = ThreadSafeDict()  
document_contents = ThreadSafeDict()
vector_stores = ThreadSafeDict()
knowledge_graphs = ThreadSafeDict()

def setup_directories() -> None:
    #initialize required directories
    directories = [
        Path('uploads'),
        Path('models'),
        Path('static'),
        Path('templates'),
        Path('knowledge_graph_data'),
        Path('knowledge_graph_data/data'),
        Path('knowledge_graph_data/visualizations'),
        Path('vector_stores')
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")

def load_document(file_path: str) -> Optional[str]:
     #load & extract text from document
    try:
        text = ""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
        elif file_extension == '.docx':
            doc = docx.Document(file_path)
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
        elif file_extension == '.csv':
            df = pd.read_csv(file_path)
            text = df.to_string()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        return text.strip() if text else None
    except Exception as e:
        logger.exception(f"Error loading document: {e}")
        return None

def chunk_text(text: str) -> List[str]:  #split text into chunks
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_text(text)

#create and save FAISS vector store
def create_vector_store(chunks: List[str], file_name: str) -> Optional[FAISS]:
    
    try:
        if not chunks:
            raise ValueError("Empty chunks provided")
        
        vector_store_dir = Path("vector_stores")
        cache_dir = Path("models")
        store_path = vector_store_dir / f"{file_name}_store"
        
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            cache_folder=str(cache_dir),
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
        )
        
        vector_store = FAISS.from_texts(
            texts=[str(chunk) for chunk in chunks],
            embedding=embeddings
        )
        
        vector_store.save_local(str(store_path))
        logger.info(f"Saved vector store to {store_path}")
        
        return vector_store
        
    except Exception as e:
        logger.exception(f"Vector store creation failed: {e}")
        return None
#process a single file and return results
def process_single_file(file, filename: str) -> Tuple[bool, dict, str]: 
    try:
        file_path = Path(app.config['UPLOAD_FOLDER']) / filename
        file.save(str(file_path))
        logger.info(f"File saved: {file_path}")
        
        text = load_document(str(file_path))
        if not text:
            raise ValueError("Failed to extract text from document")
        
        chunks = chunk_text(text)
        vector_store = create_vector_store(chunks, Path(filename).stem)
        
        if not vector_store:
            raise ValueError("Failed to create vector store")
        
        kg = create_and_save_knowledge_graph(text, Path(filename).stem)
        
        file_ext = Path(filename).suffix.lower()
        metadata = {
            'path': str(file_path),
            'size': os.path.getsize(str(file_path)),
            'uploaded_at': datetime.now().isoformat(),
            'type': file_ext[1:].upper(),
            'chunks': len(chunks),
            'vector_store': f"{Path(filename).stem}_store",
            'has_knowledge_graph': kg is not None
        }
        
        documents[filename] = metadata
        document_contents[filename] = text
        vector_stores[filename] = vector_store
        if kg:
            knowledge_graphs[filename] = kg
            
        return True, metadata, ""
        
    except Exception as e:
        error_msg = str(e)
        logger.exception(f"Processing failed for {filename}: {error_msg}")
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()
        return False, {}, error_msg

#create and save knowledge graph from text
def create_and_save_knowledge_graph(text: str, filename: str) -> Optional[KnowledgeGraph]:

    try:
        entities = extract_entities(text)
        relationships = extract_relationships(text)
        
        kg = KnowledgeGraph()
        kg.create_from_data(entities, relationships)
        
        kg_data_dir = Path("knowledge_graph_data/data")
        save_to_csv(entities, relationships, str(kg_data_dir / filename))
    
        viz_dir = Path("knowledge_graph_data/visualizations")
        viz_dir.mkdir(parents=True, exist_ok=True)
        kg.save_visualization(viz_dir, filename)
        
        logger.info(f"Created knowledge graph with {len(entities)} entities and {len(relationships)} relationships")
        return kg
    
    except Exception as e:
        logger.error(f"Knowledge graph creation failed: {e}")
        return None

@app.route('/') #Render main page
def index():
    return render_template('index.html', documents=list(documents.keys()))

@app.route('/features') #Render features page
def features():
    return render_template('features.html')

@app.route('/about') #Render about page
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST']) #Handle multiple file uploads
def upload_file():
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    files = request.files.getlist('file')
    if not files or all(file.filename == '' for file in files):
        return jsonify({"error": "No files selected"}), 400

    results = []
    errors = []
    allowed_extensions = {'.pdf', '.docx', '.csv'}

    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_file = {}
        
        for file in files:
            file_ext = Path(file.filename).suffix.lower()
            
            if file_ext not in allowed_extensions:
                errors.append({
                    "filename": file.filename,
                    "error": f"Unsupported file type. Allowed: {', '.join(allowed_extensions)}"
                })
                continue
                
            file.seek(0, 2)
            size = file.tell()
            file.seek(0)
            
            if size > app.config['MAX_CONTENT_LENGTH']:
                errors.append({
                    "filename": file.filename,
                    "error": "File too large. Maximum size is 16MB"
                })
                continue
            
            filename = secure_filename(file.filename)
            future = executor.submit(process_single_file, file, filename)
            future_to_file[future] = filename

        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                success, metadata, error_msg = future.result()
                if success:
                    results.append({
                        "filename": filename,
                        "status": "success",
                        "metadata": metadata
                    })
                else:
                    errors.append({
                        "filename": filename,
                        "error": error_msg
                    })
            except Exception as e:
                errors.append({
                    "filename": filename,
                    "error": str(e)
                })

    response = {
        "processed": results,
        "documents": list(documents.keys())
    }
    
    if errors:
        response["errors"] = errors
        
    return jsonify(response)

@app.route('/query', methods=['POST'])  #Process document queries
def query_document():
    try:
        data = request.json
        selected_docs = data.get('documents', [])
        query = data.get('query', '').strip()
        
        if not selected_docs:
            return jsonify({"error": "Please select at least one document"}), 400
        if not query:
            return jsonify({"error": "Please enter a question"}), 400

        if any(doc not in vector_stores for doc in selected_docs):
            return jsonify({"error": "One or more documents not found"}), 404

        if query.lower() in ['hi', 'hello', 'hey']:
            return jsonify({
                "answer": f"Hello! I'm analyzing {len(selected_docs)} document(s). How can I help you?"
            })

       
        query_entities = extract_entities(query)  #query Processing
        all_results = []
        entity_relationships = {}
        
        for doc_name in selected_docs:  #vector search
            vector_store = vector_stores[doc_name]
            docs = vector_store.similarity_search(query, k=2)
            all_results.extend([(doc, doc_name) for doc in docs])
            
            #KG search
            if doc_name in knowledge_graphs:
                kg = knowledge_graphs[doc_name]
                #search direct query entities
                for entity, entity_type in query_entities:
                    if entity not in entity_relationships:
                        entity_relationships[entity] = set()
                    results = kg.query_graph(entity)
                    if results:
                        entity_relationships[entity].update([
                            f"{result['entity']} {result['relationship']} {result['related_entity']}"
                            for result in results
                        ])
                
                #search related entities in context
                doc_entities = extract_entities(" ".join(doc.page_content for doc, _ in all_results))
                for entity, _ in doc_entities:
                    if entity not in entity_relationships:
                        results = kg.query_graph(entity)
                        if results:
                            entity_relationships[entity] = set([
                                f"{result['entity']} {result['relationship']} {result['related_entity']}"
                                for result in results
                            ])

        #building Context
        context_parts = []
        
        #document content
        doc_contexts = [
            f"From {doc_name}: {doc.page_content}" 
            for doc, doc_name in sorted(all_results, 
                key=lambda x: x[0].metadata.get('score', 0), 
                reverse=True)[:4]
        ]
        if doc_contexts:
            context_parts.append("Document Content:")
            context_parts.extend(doc_contexts)

        #entity relationships
        if entity_relationships:
            context_parts.append("\nEntity Relationships:")
            for entity, relationships in entity_relationships.items():
                if relationships:
                    context_parts.append(f"\n{entity}:")
                    context_parts.extend([f"- {rel}" for rel in relationships])

        context = "\n".join(context_parts)

        #response genration
        try:
            llm = Ollama(
                model="llama2",
                temperature=0.7,
                base_url="http://localhost:11434"
            )
            
            prompt = f"""You are an AI assistant analyzing documents using both content and relationship analysis.
            
            Context and Relationships:
            {context}

            Question: {query}

            Instructions: 
            Provide a complete answer that covers all the essential points in a logical and flowing narrative. 
            Start by highlighting the most valuable insights and then transition smoothly into the additional details, drawing insights from all relevant information and sections, without abrupt breaks or unnecessary repetition.
            Use a confident and knowledgeable tone throughout the response, ensuring that every key detail is addressed. 
            If there is any missing information, state it clearly without deviating from the focus of the question. 
            Extract answers directly from the document's content, referencing entity relationships only when absolutely necessary. 
            Avoid phrases such as "based on," "according to," or similar expressions, and ensure that the narrative remains natural and conversational without reiterating the question or repeating the same information.
            Answer the question clearly, using a natural and flowing narrative style, as though you're engaging in a conversation.
            
            Aim for a comprehensive yet natural response that directly answers the question."""
            
            answer = llm.invoke(prompt)
            
            if not answer or answer.strip() == "":
                return jsonify({
                    "answer": "I couldn't find specific information about that. Please try rephrasing your question."
                })
            
            return jsonify({
                "answer": answer,
                "has_relationships": bool(entity_relationships)
            })
            
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return jsonify({
                "error": "Failed to connect to Ollama. Please ensure the service is running."
            }), 503

    except Exception as e:
        logger.exception(f"Query processing failed: {e}")
        return jsonify({
            "error": "An error occurred while processing your query. Please try again."
        }), 500

if __name__ == '__main__':
    setup_directories()
    app.run(debug=True)