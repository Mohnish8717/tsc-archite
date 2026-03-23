"""Knowledge graph data models: entities, relationships, and graph structure."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class RelationshipType(str, Enum):
    REQUESTS = "REQUESTS"
    CAUSES = "CAUSES"
    IMPACTS = "IMPACTS"
    DEPENDS_ON = "DEPENDS_ON"
    CONFLICTS_WITH = "CONFLICTS_WITH"
    MENTIONED_WITH = "MENTIONED_WITH"


class EvidenceQuality(str, Enum):
    DIRECT = "direct"
    INFERRED = "inferred"
    QUANTIFIED = "quantified"


class RelationshipStrength(str, Enum):
    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"


class GraphEntity(BaseModel):
    """A node in the knowledge graph."""

    id: str
    name: str
    type: str  # PRODUCT, PERSON, ORG, CONSTRAINT, PAIN_POINT, METRIC
    full_name: str = ""
    mentions: int = 0
    raw_mentions: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    contexts: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    average_urgency: float = 0.0
    sentiment_distribution: dict[str, int] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GraphRelationship(BaseModel):
    """An edge in the knowledge graph."""

    id: str
    source_entity: str
    target_entity: str
    relationship_type: RelationshipType
    confidence: float = 0.0
    weight: float = 0.0
    evidence_chunks: list[str] = Field(default_factory=list)
    evidence_quality: EvidenceQuality = EvidenceQuality.INFERRED
    evidence_count: int = 0
    strength: RelationshipStrength = RelationshipStrength.MEDIUM
    created_at: datetime = Field(default_factory=datetime.utcnow)


class GraphMetadata(BaseModel):
    """Metrics about the knowledge graph."""

    total_nodes: int = 0
    total_edges: int = 0
    node_types: dict[str, int] = Field(default_factory=dict)
    avg_edges_per_node: float = 0.0
    avg_node_confidence: float = 0.0
    graph_density: float = 0.0


class KnowledgeGraph(BaseModel):
    """The complete knowledge graph: nodes, edges, indices, metadata."""

    nodes: dict[str, GraphEntity] = Field(default_factory=dict)
    edges: list[GraphRelationship] = Field(default_factory=list)
    indices: dict[str, dict[str, Any]] = Field(default_factory=dict)
    metadata: GraphMetadata = Field(default_factory=GraphMetadata)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def get_entity(self, entity_id: str) -> Optional[GraphEntity]:
        return self.nodes.get(entity_id)

    def get_neighbors(self, entity_id: str) -> list[GraphRelationship]:
        adj = self.indices.get("adjacency", {})
        edge_ids = adj.get(entity_id, [])
        return [e for e in self.edges if e.id in edge_ids]

    def get_entities_by_type(self, entity_type: str) -> list[GraphEntity]:
        return [n for n in self.nodes.values() if n.type == entity_type]
