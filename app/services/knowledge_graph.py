"""
Neo4j knowledge graph service for entity relationships and graph traversal
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

from app.models.schemas import EntityRelation
from app.utils.config import settings

logger = logging.getLogger(__name__)


class KnowledgeGraphService:
    """Service for managing knowledge graph operations with Neo4j."""
    
    def __init__(self):
        """Initialize Neo4j driver and connection."""
        self.driver = None
        self._connect()
    
    def _connect(self):
        """Establish connection to Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password)
            )
            
            # Verify connectivity
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            logger.info("Connected to Neo4j successfully")
            
        except (ServiceUnavailable, AuthError) as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            # Don't raise here to allow graceful degradation
            self.driver = None
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {e}")
            self.driver = None
    
    def close(self):
        """Close the Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def is_connected(self) -> bool:
        """Check if connected to Neo4j."""
        return self.driver is not None
    
    def create_entity(self, entity_name: str, entity_type: str, properties: Optional[Dict[str, Any]] = None) -> bool:
        """Create an entity node in the knowledge graph."""
        if not self.driver:
            logger.warning("Neo4j not connected, skipping entity creation")
            return False
        
        try:
            with self.driver.session() as session:
                props = properties or {}
                props.update({"name": entity_name, "type": entity_type})
                
                query = """
                MERGE (e:Entity {name: $name, type: $type})
                SET e += $properties
                RETURN e
                """
                
                session.run(query, name=entity_name, type=entity_type, properties=props)
                logger.debug(f"Created/updated entity: {entity_name} ({entity_type})")
                return True
                
        except Exception as e:
            logger.error(f"Failed to create entity {entity_name}: {e}")
            return False
    
    def create_relationship(
        self, 
        source_entity: str, 
        target_entity: str, 
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create a relationship between two entities."""
        if not self.driver:
            logger.warning("Neo4j not connected, skipping relationship creation")
            return False
        
        try:
            with self.driver.session() as session:
                props = properties or {}
                
                query = """
                MATCH (a:Entity {name: $source})
                MATCH (b:Entity {name: $target})
                MERGE (a)-[r:RELATED {type: $rel_type}]->(b)
                SET r += $properties
                RETURN r
                """
                
                result = session.run(
                    query, 
                    source=source_entity, 
                    target=target_entity, 
                    rel_type=relationship_type,
                    properties=props
                )
                
                if result.single():
                    logger.debug(f"Created relationship: {source_entity} -[{relationship_type}]-> {target_entity}")
                    return True
                else:
                    logger.warning(f"Could not create relationship between {source_entity} and {target_entity}")
                    return False
                
        except Exception as e:
            logger.error(f"Failed to create relationship: {e}")
            return False
    
    def get_related_entities(self, entity_name: str, max_depth: int = 2) -> List[EntityRelation]:
        """Get entities related to the given entity within specified depth."""
        if not self.driver:
            logger.warning("Neo4j not connected, returning empty relations")
            return []
        
        try:
            with self.driver.session() as session:
                query = """
                MATCH (start:Entity {name: $entity_name})
                MATCH (start)-[r:RELATED*1..$max_depth]-(related:Entity)
                RETURN DISTINCT related.name as entity, 
                       related.type as entity_type,
                       r[0].type as relationship,
                       length(r) as depth
                ORDER BY depth, related.name
                LIMIT 20
                """
                
                result = session.run(query, entity_name=entity_name, max_depth=max_depth)
                
                relations = []
                for record in result:
                    relation = EntityRelation(
                        entity=record["entity"],
                        relationship=record["relationship"] or "RELATED_TO",
                        context=f"Entity type: {record['entity_type']}, Depth: {record['depth']}"
                    )
                    relations.append(relation)
                
                logger.debug(f"Found {len(relations)} related entities for {entity_name}")
                return relations
                
        except Exception as e:
            logger.error(f"Failed to get related entities for {entity_name}: {e}")
            return []
    
    def find_entities_by_keywords(self, keywords: List[str]) -> List[EntityRelation]:
        """Find entities matching the given keywords."""
        if not self.driver:
            logger.warning("Neo4j not connected, returning empty results")
            return []
        
        try:
            with self.driver.session() as session:
                # Create regex pattern for keyword matching
                keyword_pattern = "|".join([f"(?i).*{kw}.*" for kw in keywords])
                
                query = """
                MATCH (e:Entity)
                WHERE e.name =~ $pattern
                RETURN e.name as entity, e.type as entity_type
                ORDER BY e.name
                LIMIT 10
                """
                
                result = session.run(query, pattern=keyword_pattern)
                
                entities = []
                for record in result:
                    entity = EntityRelation(
                        entity=record["entity"],
                        relationship="KEYWORD_MATCH",
                        context=f"Entity type: {record['entity_type']}"
                    )
                    entities.append(entity)
                
                logger.debug(f"Found {len(entities)} entities matching keywords: {keywords}")
                return entities
                
        except Exception as e:
            logger.error(f"Failed to find entities by keywords: {e}")
            return []
    
    def add_document_entities(self, doc_id: str, entities: List[Dict[str, Any]]) -> int:
        """Add entities extracted from a document to the knowledge graph."""
        if not self.driver:
            logger.warning("Neo4j not connected, skipping document entities")
            return 0
        
        added_count = 0
        
        try:
            # Create document node
            if self.create_entity(doc_id, "DOCUMENT"):
                added_count += 1
            
            # Create entity nodes and relationships to document
            for entity_data in entities:
                entity_name = entity_data.get("name")
                entity_type = entity_data.get("type", "CONCEPT")
                
                if not entity_name:
                    continue
                
                # Create entity
                if self.create_entity(entity_name, entity_type, entity_data):
                    added_count += 1
                
                # Create relationship to document
                if self.create_relationship(doc_id, entity_name, "CONTAINS"):
                    added_count += 1
            
            logger.info(f"Added {added_count} entities/relationships for document {doc_id}")
            return added_count
            
        except Exception as e:
            logger.error(f"Failed to add document entities: {e}")
            return added_count
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get basic statistics about the knowledge graph."""
        if not self.driver:
            return {"status": "disconnected"}
        
        try:
            with self.driver.session() as session:
                # Count nodes
                node_result = session.run("MATCH (n) RETURN count(n) as node_count")
                node_count = node_result.single()["node_count"]
                
                # Count relationships
                rel_result = session.run("MATCH ()-[r]-() RETURN count(r) as rel_count")
                rel_count = rel_result.single()["rel_count"]
                
                # Count entity types
                type_result = session.run("""
                    MATCH (n:Entity) 
                    RETURN n.type as entity_type, count(n) as count 
                    ORDER BY count DESC
                """)
                
                entity_types = {}
                for record in type_result:
                    entity_types[record["entity_type"]] = record["count"]
                
                return {
                    "status": "connected",
                    "nodes": node_count,
                    "relationships": rel_count,
                    "entity_types": entity_types
                }
                
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {"status": "error", "error": str(e)}


# Global knowledge graph service instance
kg_service = KnowledgeGraphService()
