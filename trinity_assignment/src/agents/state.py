

from typing import List, TypedDict, Annotated, Optional, Dict, Any
from langchain_core.documents import Document
import operator

class AgentState(TypedDict):
    
    query: str 
    plan: Optional[Dict[str, Any]]
    optimized_queries: List[str] 
    documents: Annotated[List[Document], operator.add]
    retriever_content: str 
    retrieval_success: bool
    generation: str
    revision_notes: str
    retry_count: int 
    confidence_score: float


