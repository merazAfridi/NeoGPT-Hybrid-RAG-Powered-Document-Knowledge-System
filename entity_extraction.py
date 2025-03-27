import spacy
import pandas as pd
from typing import List, Tuple, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

#extract named entities from text
def extract_entities(text: str) -> List[Tuple[str, str]]:
    try:
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        logger.info(f"Extracted {len(entities)} entities")
        return entities
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return []

#extract relationships between entities
def extract_relationships(text: str) -> List[Tuple[str, str, str]]:
    try:
        doc = nlp(text)
        relationships = []
        
        for sent in doc.sents:
            sent_doc = nlp(sent.text)
            
            #find subject-verb-object patterns
            for token in sent_doc:
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    subject = token.text
                    verb = token.head.text
                    
                    #find object
                    for child in token.head.children:
                        if child.dep_ in ["dobj", "pobj"]:
                            obj = child.text
                            relationships.append((subject, verb, obj))
        
        logger.info(f"Extracted {len(relationships)} relationships")
        return relationships
    except Exception as e:
        logger.error(f"Relationship extraction failed: {e}")
        return []

def process_document(file_path: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
    #process document to extract entities & relationships
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        entities = extract_entities(text)
        relationships = extract_relationships(text)
        
        logger.info(f"Processed document: {file_path}")
        logger.info(f"Found {len(entities)} entities and {len(relationships)} relationships")
        return entities, relationships
    
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        return [], []

def save_to_csv(entities: List[Tuple[str, str]], 
                relationships: List[Tuple[str, str, str]], 
                output_file: str) -> None:

    ##Save extracted information to CSV files
    try:  
        output_dir = Path("knowledge_graph_data/data")
        output_dir.mkdir(parents=True, exist_ok=True)
       
        filename = Path(output_file).stem  #clean filename

        entities_path = output_dir / f"{filename}_entities.csv"
        relationships_path = output_dir / f"{filename}_relationships.csv"
        
        entities_df = pd.DataFrame(entities, columns=["Entity", "Type"])  #save entities
        entities_df.to_csv(entities_path, index=False)
        
        #  relationships save
        relationships_df = pd.DataFrame(relationships, columns=["Entity1", "Relationship", "Entity2"])
        relationships_df.to_csv(relationships_path, index=False)
        
        logger.info(f"Saved entities to: {entities_path}")
        logger.info(f"Saved relationships to: {relationships_path}")
    
    except Exception as e:
        logger.error(f"Failed to save data: {e}")
        raise

def verify_directories(): #Verify required directories exist
    try:
        required_dirs = [
            Path("knowledge_graph_data"),
            Path("knowledge_graph_data/data")
        ]
        
        for directory in required_dirs:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Verified directory exists: {directory}")
            
    except Exception as e:
        logger.error(f"Directory verification failed: {e}")
        raise

if __name__ == "__main__":
    try:
        verify_directories() #verify directories exist

        #process document
        file_path = "path_to_your_document.txt" 
        entities, relationships = process_document(file_path)

        save_to_csv(entities, relationships, "knowledge_graph")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")