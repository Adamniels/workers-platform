"""Pydantic models for .NET MemoryContextV1 JSON (camelCase)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ProfileFactV1(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    source: str = ""
    text: str = ""
    authority_weight: float = Field(0, alias="authorityWeight")
    rank_score: float = Field(0, alias="rankScore")


class ActiveGoalV1(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    goal: str = ""
    authority_weight: float = Field(0, alias="authorityWeight")
    rank_score: float = Field(0, alias="rankScore")


class RelevantProjectV1(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = ""
    external_id: str | None = Field(None, alias="externalId")
    rank_score: float = Field(0, alias="rankScore")


class SemanticMemoryContextV1(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int = 0
    key: str = ""
    claim: str = ""
    domain: str | None = None
    confidence: float = 0
    authority_weight: float = Field(0, alias="authorityWeight")
    status: str = ""
    updated_at: str | None = Field(None, alias="updatedAt")
    rank_score: float = Field(0, alias="rankScore")
    evidence_link_count: int = Field(0, alias="evidenceLinkCount")


class ProceduralRuleContextV1(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int = 0
    workflow_type: str = Field("", alias="workflowType")
    rule_name: str = Field("", alias="ruleName")
    rule_content: str = Field("", alias="ruleContent")
    priority: int = 0
    version: int = 0
    status: str = ""
    source: str = ""
    authority_weight: float = Field(0, alias="authorityWeight")
    rank_score: float = Field(0, alias="rankScore")


class MemoryItemVectorRecallV1(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    memory_item_id: int = Field(0, alias="memoryItemId")
    chunk_index: int = Field(0, alias="chunkIndex")
    memory_type: str = Field("", alias="memoryType")
    title: str = ""
    content_preview: str = Field("", alias="contentPreview")
    cosine_similarity: float = Field(0, alias="cosineSimilarity")
    authority_weight: float = Field(0, alias="authorityWeight")
    rank_score: float = Field(0, alias="rankScore")
    embedding_model_key: str = Field("", alias="embeddingModelKey")
    is_document_evidence: bool = Field(False, alias="isDocumentEvidence")
    project_id: str | None = Field(None, alias="projectId")
    domain: str | None = None
    source_type: str = Field("", alias="sourceType")


class MemoryContextV1(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    profile_facts: list[ProfileFactV1] = Field(
        default_factory=list,
        alias="profileFacts",
    )
    active_goals: list[ActiveGoalV1] = Field(
        default_factory=list,
        alias="activeGoals",
    )
    relevant_projects: list[RelevantProjectV1] = Field(
        default_factory=list,
        alias="relevantProjects",
    )
    semantic_memories: list[SemanticMemoryContextV1] = Field(
        default_factory=list,
        alias="semanticMemories",
    )
    procedural_rules: list[ProceduralRuleContextV1] = Field(
        default_factory=list,
        alias="proceduralRules",
    )
    memory_item_vector_recalls: list[MemoryItemVectorRecallV1] = Field(
        default_factory=list,
        alias="memoryItemVectorRecalls",
    )
    vector_recall_used: bool = Field(False, alias="vectorRecallUsed")
    assembly_stage: str = Field("v1-sql", alias="assemblyStage")


class GetMemoryContextV1Request(BaseModel):
    """Request body for POST .../memory/context (public or internal)."""

    model_config = ConfigDict(populate_by_name=True)

    user_id: int | None = Field(None, alias="userId")
    task_description: str | None = Field(None, alias="taskDescription")
    workflow_type: str | None = Field(None, alias="workflowType")
    project_id: str | None = Field(None, alias="projectId")
    domain: str | None = None
    include_vector_recall: bool | None = Field(None, alias="includeVectorRecall")
