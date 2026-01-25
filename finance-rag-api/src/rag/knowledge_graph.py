# -*- coding: utf-8 -*-
"""
Knowledge Graph 모듈

기업 관계, 산업 구조 등을 그래프로 모델링하여 RAG를 강화합니다.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Set

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


class EntityType(Enum):
    """엔티티 유형"""
    COMPANY = "company"
    PERSON = "person"
    INDUSTRY = "industry"
    PRODUCT = "product"
    EVENT = "event"
    METRIC = "metric"


class RelationType(Enum):
    """관계 유형"""
    SUBSIDIARY = "subsidiary"
    PARENT = "parent"
    AFFILIATE = "affiliate"
    COMPETITOR = "competitor"
    SUPPLIER = "supplier"
    CUSTOMER = "customer"
    PARTNER = "partner"
    INVESTOR = "investor"
    CEO = "ceo"
    EXECUTIVE = "executive"
    BOARD_MEMBER = "board_member"
    BELONGS_TO = "belongs_to"
    PRODUCES = "produces"


@dataclass
class Entity:
    """엔티티 (노드)"""
    entity_id: str
    name: str
    entity_type: EntityType
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "properties": self.properties,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        return cls(
            entity_id=data["entity_id"],
            name=data["name"],
            entity_type=EntityType(data["entity_type"]),
            properties=data.get("properties", {}),
        )


@dataclass
class Relation:
    """관계 (엣지)"""
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Relation":
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=RelationType(data["relation_type"]),
            properties=data.get("properties", {}),
            weight=data.get("weight", 1.0),
        )


class FinanceKnowledgeGraph:
    """금융 지식 그래프"""

    def __init__(self):
        self._entities: dict[str, Entity] = {}
        self._relations: list[Relation] = []
        self._graph = nx.DiGraph() if NETWORKX_AVAILABLE else None
        self._created_at = datetime.now()

    @property
    def num_entities(self) -> int:
        return len(self._entities)

    @property
    def num_relations(self) -> int:
        return len(self._relations)

    @property
    def stats(self) -> dict[str, Any]:
        entity_counts = {}
        for entity in self._entities.values():
            t = entity.entity_type.value
            entity_counts[t] = entity_counts.get(t, 0) + 1

        return {
            "num_entities": self.num_entities,
            "num_relations": self.num_relations,
            "entity_counts": entity_counts,
        }

    def add_entity(self, entity: Entity) -> None:
        self._entities[entity.entity_id] = entity
        if self._graph is not None:
            self._graph.add_node(entity.entity_id, name=entity.name,
                                 entity_type=entity.entity_type.value)

    def add_relation(self, relation: Relation) -> None:
        if relation.source_id not in self._entities:
            return
        if relation.target_id not in self._entities:
            return
        self._relations.append(relation)
        if self._graph is not None:
            self._graph.add_edge(relation.source_id, relation.target_id,
                                 relation_type=relation.relation_type.value)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self._entities.get(entity_id)

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        for entity in self._entities.values():
            if entity.name == name:
                return entity
        return None

    def get_entities_by_type(self, entity_type: EntityType) -> list[Entity]:
        return [e for e in self._entities.values() if e.entity_type == entity_type]

    def get_relations(self, source_id: Optional[str] = None,
                      target_id: Optional[str] = None) -> list[Relation]:
        results = []
        for relation in self._relations:
            if source_id and relation.source_id != source_id:
                continue
            if target_id and relation.target_id != target_id:
                continue
            results.append(relation)
        return results

    def get_neighbors(self, entity_id: str, direction: str = "both") -> list[Entity]:
        neighbor_ids: Set[str] = set()
        for relation in self._relations:
            if direction in ("out", "both") and relation.source_id == entity_id:
                neighbor_ids.add(relation.target_id)
            if direction in ("in", "both") and relation.target_id == entity_id:
                neighbor_ids.add(relation.source_id)
        return [self._entities[nid] for nid in neighbor_ids if nid in self._entities]

    def get_path(self, source_id: str, target_id: str, max_depth: int = 3) -> list[list[str]]:
        if self._graph is not None and NETWORKX_AVAILABLE:
            try:
                return list(nx.all_simple_paths(self._graph, source_id, target_id, cutoff=max_depth))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                return []
        return self._bfs_paths(source_id, target_id, max_depth)

    def _bfs_paths(self, source_id: str, target_id: str, max_depth: int) -> list[list[str]]:
        if source_id not in self._entities or target_id not in self._entities:
            return []
        paths = []
        queue = [[source_id]]
        while queue:
            path = queue.pop(0)
            if len(path) > max_depth + 1:
                continue
            if path[-1] == target_id:
                paths.append(path)
                continue
            for relation in self._relations:
                if relation.source_id == path[-1] and relation.target_id not in path:
                    queue.append(path + [relation.target_id])
        return paths

    def to_context(self, entity_id: str, depth: int = 1) -> str:
        entity = self.get_entity(entity_id)
        if not entity:
            return ""
        lines = [f"## {entity.name} ({entity.entity_type.value})"]
        for key, value in entity.properties.items():
            lines.append(f"- {key}: {value}")
        relations = self.get_relations(source_id=entity_id)
        if relations:
            lines.append("\n### 관련 정보")
            for relation in relations[:10]:
                target = self.get_entity(relation.target_id)
                if target:
                    lines.append(f"- {relation.relation_type.value}: {target.name}")
        return "\n".join(lines)

    def save(self, path: str) -> None:
        data = {
            "entities": [e.to_dict() for e in self._entities.values()],
            "relations": [r.to_dict() for r in self._relations],
            "created_at": self._created_at.isoformat(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "FinanceKnowledgeGraph":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        graph = cls()
        graph._created_at = datetime.fromisoformat(data["created_at"])
        for entity_data in data["entities"]:
            graph.add_entity(Entity.from_dict(entity_data))
        for relation_data in data["relations"]:
            graph.add_relation(Relation.from_dict(relation_data))
        return graph


class KoreanFinanceKGBuilder:
    """한국 금융 Knowledge Graph 빌더"""

    MAJOR_COMPANIES = [
        {"id": "samsung_elec", "name": "삼성전자", "industry": "semiconductor", "market_cap": "400조"},
        {"id": "sk_hynix", "name": "SK하이닉스", "industry": "semiconductor", "market_cap": "150조"},
        {"id": "lg_energy", "name": "LG에너지솔루션", "industry": "battery", "market_cap": "100조"},
        {"id": "hyundai_motor", "name": "현대차", "industry": "auto", "market_cap": "60조"},
        {"id": "naver", "name": "네이버", "industry": "internet", "market_cap": "50조"},
        {"id": "kakao", "name": "카카오", "industry": "internet", "market_cap": "30조"},
    ]

    INDUSTRIES = [
        {"id": "semiconductor", "name": "반도체"},
        {"id": "battery", "name": "배터리"},
        {"id": "auto", "name": "자동차"},
        {"id": "internet", "name": "인터넷"},
    ]

    COMPETITORS = [
        ("samsung_elec", "sk_hynix"),
        ("naver", "kakao"),
    ]

    def build(self) -> FinanceKnowledgeGraph:
        kg = FinanceKnowledgeGraph()

        for industry in self.INDUSTRIES:
            kg.add_entity(Entity(
                entity_id=industry["id"],
                name=industry["name"],
                entity_type=EntityType.INDUSTRY,
            ))

        for company in self.MAJOR_COMPANIES:
            kg.add_entity(Entity(
                entity_id=company["id"],
                name=company["name"],
                entity_type=EntityType.COMPANY,
                properties={"market_cap": company["market_cap"]},
            ))
            kg.add_relation(Relation(
                source_id=company["id"],
                target_id=company["industry"],
                relation_type=RelationType.BELONGS_TO,
            ))

        for source_id, target_id in self.COMPETITORS:
            kg.add_relation(Relation(source_id=source_id, target_id=target_id,
                                     relation_type=RelationType.COMPETITOR))
            kg.add_relation(Relation(source_id=target_id, target_id=source_id,
                                     relation_type=RelationType.COMPETITOR))

        return kg


class KGEnhancedRAG:
    """Knowledge Graph 강화 RAG"""

    def __init__(self, knowledge_graph: FinanceKnowledgeGraph):
        self.kg = knowledge_graph

    def expand_query_with_kg(self, query: str, max_expansions: int = 5) -> dict[str, Any]:
        mentioned_entities = []
        for entity in self.kg._entities.values():
            if entity.name in query:
                mentioned_entities.append(entity)

        related_entities = []
        for entity in mentioned_entities:
            neighbors = self.kg.get_neighbors(entity.entity_id)
            related_entities.extend(neighbors[:max_expansions])

        expanded_terms = [e.name for e in related_entities]
        return {
            "original_query": query,
            "mentioned_entities": [e.name for e in mentioned_entities],
            "related_entities": expanded_terms[:max_expansions],
            "expanded_query": query + " " + " ".join(expanded_terms[:3]),
        }

    def get_context_from_kg(self, query: str, depth: int = 1) -> str:
        contexts = []
        for entity in self.kg._entities.values():
            if entity.name in query:
                context = self.kg.to_context(entity.entity_id, depth)
                if context:
                    contexts.append(context)
        return "\n\n".join(contexts)

    def find_connections(self, entity1_name: str, entity2_name: str) -> Optional[str]:
        entity1 = self.kg.get_entity_by_name(entity1_name)
        entity2 = self.kg.get_entity_by_name(entity2_name)
        if not entity1 or not entity2:
            return None
        paths = self.kg.get_path(entity1.entity_id, entity2.entity_id)
        if not paths:
            return f"{entity1_name}과 {entity2_name} 간의 직접적인 연결을 찾지 못했습니다."
        shortest = min(paths, key=len)
        path_names = [self.kg.get_entity(eid).name for eid in shortest if self.kg.get_entity(eid)]
        return f"{entity1_name}과 {entity2_name}의 연결: {' -> '.join(path_names)}"


def build_korean_finance_kg() -> FinanceKnowledgeGraph:
    builder = KoreanFinanceKGBuilder()
    return builder.build()


def get_entity_context(kg: FinanceKnowledgeGraph, entity_name: str) -> str:
    entity = kg.get_entity_by_name(entity_name)
    if entity:
        return kg.to_context(entity.entity_id)
    return ""
