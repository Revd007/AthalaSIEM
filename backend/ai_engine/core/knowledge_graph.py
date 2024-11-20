import networkx as nx
from typing import Dict, List, Any, Optional
import torch
import numpy as np

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.embedding_dim = 256
        self.node_embeddings = {}
        
    def update(self, patterns: List[Dict[str, Any]], feedback: Optional[Dict[str, Any]] = None):
        """Update knowledge graph with new patterns"""
        for pattern in patterns:
            # Create node for pattern
            node_id = self._create_node_id(pattern)
            
            # Add node if not exists
            if not self.graph.has_node(node_id):
                self.graph.add_node(
                    node_id,
                    embedding=pattern['centroid'],
                    frequency=pattern['frequency']
                )
                
            # Update node attributes
            self.graph.nodes[node_id]['frequency'] += pattern['frequency']
            
            # Add edges to related patterns
            self._add_pattern_relationships(node_id, pattern)
            
        # Update embeddings
        self._update_embeddings()
        
        # Incorporate feedback if available
        if feedback:
            self._incorporate_feedback(feedback)
    
    def _create_node_id(self, pattern: Dict[str, Any]) -> str:
        """Create unique identifier for pattern"""
        return hash(pattern['centroid'].tostring())
    
    def _add_pattern_relationships(self, node_id: str, pattern: Dict[str, Any]):
        """Add edges between related patterns"""
        # Find similar patterns
        for other_node in self.graph.nodes:
            if other_node == node_id:
                continue
                
            similarity = self._calculate_similarity(
                pattern['centroid'],
                self.graph.nodes[other_node]['embedding']
            )
            
            if similarity > 0.8:  # High similarity threshold
                self.graph.add_edge(node_id, other_node, weight=similarity)
    
    def _calculate_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """Calculate cosine similarity between embeddings"""
        return torch.nn.functional.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0)
        ).item()
    
    def _update_embeddings(self):
        """Update node embeddings using graph structure"""
        # Use node2vec or similar algorithm
        node2vec = node2vec(
            self.graph,
            dimensions=self.embedding_dim,
            walk_length=30,
            num_walks=200,
            workers=4
        )
        
        # Train embeddings
        model = node2vec.fit(window=10, min_count=1)
        
        # Update node embeddings
        for node in self.graph.nodes:
            self.node_embeddings[node] = torch.tensor(
                model.wv[str(node)],
                dtype=torch.float32
            )
    
    def _incorporate_feedback(self, feedback: Dict[str, Any]):
        """Incorporate user feedback into knowledge graph"""
        if 'correct' in feedback:
            node_id = feedback['pattern_id']
            if self.graph.has_node(node_id):
                # Increase importance of correct patterns
                self.graph.nodes[node_id]['importance'] = \
                    self.graph.nodes[node_id].get('importance', 1.0) * 1.1
                
        if 'incorrect' in feedback:
            node_id = feedback['pattern_id']
            if self.graph.has_node(node_id):
                # Decrease importance of incorrect patterns
                self.graph.nodes[node_id]['importance'] = \
                    self.graph.nodes[node_id].get('importance', 1.0) * 0.9