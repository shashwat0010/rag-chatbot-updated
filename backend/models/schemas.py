from typing import List, Optional

from pydantic import BaseModel, Field


class Citation(BaseModel):
    title: str
    journal: str
    year: Optional[int] = None
    pubmed_url: str
    pmid: str
    authors: Optional[str] = None


class PaperMetadata(BaseModel):
    pmid: str
    title: str
    abstract: str
    journal: str
    year: Optional[int] = None
    authors: Optional[str] = None
    pubmed_url: str
    doi: Optional[str] = None


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000, description="Medical research question")
    max_papers: Optional[int] = Field(None, ge=1, le=30)


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence_note: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    insufficient_evidence: bool = False
    sources_searched: List[str] = Field(default_factory=lambda: ["PubMed"])
    confidence_label: Optional[str] = None


class SearchPapersRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)
    max_results: Optional[int] = Field(None, ge=1, le=30)


class SearchPapersResponse(BaseModel):
    papers: List[PaperMetadata]
    total: int
    query: str


from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_configured: bool


class UserRegister(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="User password")
    name: Optional[str] = None


class UserLogin(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserResponse"


class UserResponse(BaseModel):
    id: int
    email: str
    name: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class SearchHistoryItem(BaseModel):
    id: int
    query: str
    response: str
    confidence_score: Optional[str] = None
    evidence_count: int = 0
    created_at: str

    model_config = ConfigDict(from_attributes=True)


