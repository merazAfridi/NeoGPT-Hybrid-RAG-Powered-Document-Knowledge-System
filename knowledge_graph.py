import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use('Agg')  #set non interactive backend
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Set
import logging
from pathlib import Path
import colorsys
import threading
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self): #initialize knowledge graph with thread safety
        self.G = nx.Graph()
        self.entity_types: Dict[str, str] = {}
        self._lock = threading.Lock()
        
    def create_from_data(self, 
                        entities: List[Tuple[str, str]], 
                        relationships: List[Tuple[str, str, str]]) -> bool:
        #create knowledge graph from entities& relationships
        try:
            with self._lock:
                self.G.clear() #clear existing graph
                self.entity_types.clear()
                #add entities with their types
                for entity, entity_type in entities: 
                    if entity and entity_type: 
                        self.G.add_node(entity, type=entity_type)
                        self.entity_types[entity] = entity_type
                
                #add relationships
                for entity1, relationship, entity2 in relationships:
                    if entity1 and entity2 and relationship: 
                        if entity1 in self.G and entity2 in self.G:
                            self.G.add_edge(entity1, entity2, relationship=relationship)
                
                logger.info(f"Created knowledge graph with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges")
                return True
        
        except Exception as e:
            logger.error(f"Failed to create knowledge graph: {e}")
            return False

    def generate_distinct_colors(self, n: int) -> List[str]:
        #generate distinct colors for entities
        try:
            colors = []
            golden_ratio_conjugate = 0.618033988749895
            hue = 0.1
            
            for _ in range(n):
                hue = (hue + golden_ratio_conjugate) % 1
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
                colors.append(f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}')
            
            return colors
            
        except Exception as e:
            logger.error(f"Color generation failed: {e}")
            return ['#1f77b4'] * n  
    
    def load_from_csv(self, entities_file: str, relationships_file: str) -> bool:
        #load KG from CSV 
        try:
            with self._lock:
                entities_df = pd.read_csv(entities_file) #load entities
                valid_entities = entities_df.dropna()
                
                for _, row in valid_entities.iterrows():
                    self.G.add_node(row["Entity"], type=row["Type"])
                    self.entity_types[row["Entity"]] = row["Type"]
                
                relationships_df = pd.read_csv(relationships_file)
                valid_relationships = relationships_df.dropna()
                
                for _, row in valid_relationships.iterrows():
                    if row["Entity1"] in self.G and row["Entity2"] in self.G:
                        self.G.add_edge(
                            row["Entity1"], 
                            row["Entity2"], 
                            relationship=row["Relationship"]
                        )
                
                logger.info(f"Loaded knowledge graph with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges")
                return True
        
        except Exception as e:
            logger.error(f"Failed to load knowledge graph from CSV: {e}")
            return False
    
    def get_entity_neighborhood(self, entity: str, depth: int = 1) -> Optional[nx.Graph]:
        #get subgraph around specific entity with specified depth
        try:
            with self._lock:
                if entity not in self.G:
                    return None
                
                nodes: Set[str] = {entity}
                current_nodes: Set[str] = {entity}
                
                for _ in range(depth):
                    next_nodes: Set[str] = set()
                    for node in current_nodes:
                        next_nodes.update(self.G.neighbors(node))
                    nodes.update(next_nodes)
                    current_nodes = next_nodes
                
                return self.G.subgraph(nodes)
        
        except Exception as e:
            logger.error(f"Failed to get entity neighborhood: {e}")
            return None
    
    def save_visualization(self, output_path: Path, filename: str) -> Optional[Path]:
        #save visualization of KG
        try:
            with self._lock:
                #output directory creation if it doesn't exist
                output_path.mkdir(parents=True, exist_ok=True)
                png_path = output_path / f"{filename}_graph.png"

                #generate color map for entity types
                unique_types = list(set(self.entity_types.values()))
                color_map = dict(zip(unique_types, self.generate_distinct_colors(len(unique_types))))

                plt.figure(figsize=(16, 12))
                pos = nx.spring_layout(self.G, k=1.5, iterations=50)
                
                #draw nodes with custom coloring
                for node_type in unique_types:
                    node_list = [node for node in self.G.nodes() 
                               if self.entity_types.get(node) == node_type]
                    if node_list:  #draw nodes with specific type
                        nx.draw_networkx_nodes(self.G, pos,
                                         nodelist=node_list,
                                         node_color=color_map[node_type],
                                         node_size=1800,
                                         alpha=0.6,
                                         label=node_type)

               #Draw edges
                edge_labels = nx.get_edge_attributes(self.G, 'relationship')
                nx.draw_networkx_edge_labels(self.G, pos, 
                                           edge_labels=edge_labels,
                                           font_size=9)
                nx.draw_networkx_edges(self.G, pos,
                                     edge_color='gray',
                                     width=1.2,
                                     alpha=0.6,
                                     arrows=True,
                                     arrowsize=22)

                #Draw labels
                nx.draw_networkx_labels(self.G, pos,
                                      font_size=11,
                                      font_weight='bold')

                #Set title and legend
                plt.title(f"Knowledge Graph: {filename}",
                         pad=20,
                         size=16,
                         fontweight='bold')
                plt.legend(title="Entity Types",
                          title_fontsize=12,
                          fontsize=10,
                          loc='center left',
                          bbox_to_anchor=(1, 0.5))

                plt.axis('off')
                plt.tight_layout()

                plt.savefig(png_path,
                           format='png',
                           dpi=400,
                           bbox_inches='tight',
                           facecolor='white')
                plt.close()

                logger.info(f"Saved visualization to: {png_path}")
                return png_path

        except Exception as e:
            logger.error(f"Failed to save visualization: {e}")
            return None

    def query_graph(self, query_entity: str) -> List[Dict]:
        #query KG for relationships of specific entity
        try:
            with self._lock:
                if query_entity not in self.G:
                    return []
                
                results = []
                for neighbor in self.G.neighbors(query_entity):
                    edge_data = self.G.get_edge_data(query_entity, neighbor)
                    if edge_data and 'relationship' in edge_data:
                        results.append({
                            'entity': query_entity,
                            'relationship': edge_data['relationship'],
                            'related_entity': neighbor,
                            'related_entity_type': self.entity_types.get(neighbor, 'Unknown')
                        })
                
                return results
        
        except Exception as e:
            logger.error(f"Failed to query graph: {e}")
            return []

    def get_all_entities(self) -> List[Dict[str, str]]:
        #get all entities in KG
        try:
            with self._lock:
                return [
                    {'entity': entity, 'type': self.entity_types[entity]}
                    for entity in self.G.nodes()
                ]
        except Exception as e:
            logger.error(f"Failed to get entities: {e}")
            return []

    def get_all_relationships(self) -> List[Dict[str, str]]:
        #get all relationships in KG
        try:
            with self._lock:
                relationships = []
                for edge in self.G.edges(data=True):
                    relationships.append({
                        'source': edge[0],
                        'target': edge[1],
                        'relationship': edge[2].get('relationship', 'unknown')
                    })
                return relationships
        except Exception as e:
            logger.error(f"Failed to get relationships: {e}")
            return []

if __name__ == "__main__":
    #example usage
    try:
        kg = KnowledgeGraph()
        viz_path = Path("knowledge_graph_data/visualizations")
        viz_path.mkdir(parents=True, exist_ok=True)

        success = kg.load_from_csv(
            "knowledge_graph_data/knowledge_graph_entities.csv",
            "knowledge_graph_data/knowledge_graph_relationships.csv"
        )
        
        if success:
            kg.save_visualization(viz_path, "knowledge_graph")
            print("\nExample entity query:")
            entities = kg.get_all_entities()
            if entities:
                sample_entity = entities[0]['entity']
                results = kg.query_graph(sample_entity)
                print(f"Relationships for {sample_entity}:")
                for result in results:
                    print(f"- {result['entity']} {result['relationship']} {result['related_entity']}")
        
    except Exception as e:
        logger.error(f"Main execution failed: {e}")