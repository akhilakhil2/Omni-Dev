"""
Planner Agent Node for the RAG Workflow.

This module contains the logic for the Planner Agent, which classifies user intent,
selects the appropriate document sections, and generates optimized search queries 
with valid ChromaDB metadata filters using the $in operator.
"""
import os
import logging
from typing import List, Dict, Any, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from .state import AgentState
from .chromadb_utils import build_chromadb_filter

logger = logging.getLogger("PlannerAgent")

# --- Structured Planner Output ---
class PlannerPlan(BaseModel):
    query_type: Literal["comparison", "definition", "recommendation", "trade-off"]
    target_sections: List[str]
    metadata_filter: Dict[str, Any]
    optimized_queries: List[str]
    is_multi_pass: bool
    reasoning: str

# --- Keyword expansion helper ---
def expand_keywords(query: str) -> List[str]:
    """
    Generate multiple search term variations for ChromaDB.
    """
    variations = [query]
    variations.extend([f"what is {query}", f"is {query}", f"{query} definition"])
    return list(set(variations))

# --- Planner Node ---
def planner_node(state: AgentState) -> Dict[str, Any]:
    logger.info("Planner Agent activated. Analyzing query...")

    # Load API key
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY missing")

    # Initialize LLM
    model = ChatGroq(model='llama-3.3-70b-versatile', api_key=groq_key, temperature=0)
    structured_llm = model.with_structured_output(PlannerPlan)

    # System message: robust for ChromaDB filters
    system_message = system_message = """
You are the Planner Agent for a Retrieval-Augmented Generation (RAG) workflow.

Your goal is to orchestrate a retrieval strategy from the AWS RAG Guide. [cite: 3, 6]

### DOCUMENT HIERARCHY
Use these headers for mapping:
- Header_2: ['Retrieval Augmented Generation options and architectures on AWS', 'Generative AI options for querying custom documents', 'Fully managed Retrieval Augmented Generation options on AWS', 'Custom Retrieval Augmented Generation architectures on AWS', 'Choosing Retrieval Augmented Generation option on AWS', 'Conclusion', 'Document history', 'AWS Prescriptive', 'Guidance glossary']
- Header_3: ['Table of Contents', 'Intended audience', 'Objectives', 'Understanding Retrieval Augmented Generation', 'Comparing Retrieval Augmented Generation and fine-tuning', 'Use cases for Retrieval Augmented Generation', 'Knowledge bases for Amazon Bedrock', 'Amazon Q Business', 'Amazon SageMaker AI Canvas', 'Retrievers for RAG workflows', 'Generators for RAG workflows']
- Header_4: ['AWS Prescriptive Guidance: Retrieval Augmented Generation options and architectures on AWS', 'Components of production-level RAG systems', 'Data sources for knowledge bases', 'Vector databases for knowledge bases', 'Key features', 'End-user customization', 'Amazon Kendra', 'Amazon OpenSearch Service', 'Amazon Aurora PostgreSQL and pgvector', 'Amazon Neptune Analytics', 'Amazon MemoryDB', 'Amazon DocumentDB', 'Pinecone', 'MongoDB Atlas', 'Weaviate', 'Amazon Bedrock', 'SageMaker AI JumpStart']

### RESPONSIBILITIES
1. CLASSIFY: Determine user intent. Options: 'comparison', 'definition', 'recommendation', 'trade-off'. [cite: 61]
2. IDENTIFY SECTIONS: Pick the Headers relevant to the query, including all levels (Header_2, Header_3, Header_4). [cite: 62]
3. DECIDE RETRIEVAL STRATEGY: Determine if single-pass or multi-pass retrieval is needed:
   - Single-pass: one query with combined filters suffices.
   - Multi-pass: separate queries per topic/component (e.g., for comparisons between two architectures).
4. GENERATE OPTIMIZED QUERIES:
   - Expand search terms to include variations for ChromaDB, e.g., "what is {query}", "is {query}", "{query} definition".
   - Provide multiple queries if the plan requires multi-pass.
5. FUZZY FILTERING RULES (CRITICAL):
   - ChromaDB metadata filters require exact matches.
   - Format: {{"Header_N": {{"$in": ["Keyword", "**Keyword**"]}}}} [cite: 76]
6. LOGICAL OPERATORS (CHROMA DB REQUIREMENT):
   - If filtering by MORE THAN ONE Header or condition, wrap in "$and" or "$or" list.
   - BAD (causes error): {{"Header_2": {{...}}, "Header_4": {{...}}}}
   - GOOD: {{"$and": [{{"Header_2": {{...}}}}, {{"Header_4": {{...}}}}]}}
   - OR logic example: {{"$or": [{{"header_2": {{"$in": ["Header A"]}}}}, {{"header_3": {{"$in": ["Header B"]}}}}]}}

### EXAMPLES OF SAFE CHROMADB FILTERS
- Single header: {{"header": {{"$in": ["Header Name"]}}}}
- OR multiple headers: {{"$or": [{{"header_2": {{"$in": ["Header A"]}}}}, {{"header_3": {{"$in": ["Header B"]}}}}]}}
- AND multiple conditions: {{"$and": [{{"header": {{"$in": ["Header A"]}}}}, {{"service": {{"$in": ["Amazon Bedrock"]}}}}]}}

### EXTRA GUIDELINES
- Always include at least 2 variations per keyword for fuzzy matching.
- If a query mentions multiple services (e.g., Amazon Bedrock, SageMaker AI JumpStart), include each in separate $and/$or conditions as needed.
- Provide reasoning in plain text explaining why you selected the headers, operators, and queries.
- Avoid using $and/$or lists with only a single element; collapse to single condition in that case.
- Ensure compatibility with ChromaDB filters to prevent runtime errors.
"""


    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "User Query: {query}\nFeedback/Revision Notes: {revision_notes}")
    ])

    planner_chain = prompt_template | structured_llm

    try:
        plan_output = planner_chain.invoke({
            "query": state.get("query"),
            "revision_notes": state.get("revision_notes") or "Initial attempt."
        })

        # Build safe ChromaDB filter from selected headers
        conditions = [{"header": {"$in": [h]}} for h in plan_output.target_sections]
        chroma_filter = build_chromadb_filter(conditions, logic="or")

        # Expand search queries
        expanded_queries = []
        for q in plan_output.optimized_queries:
            expanded_queries.extend(expand_keywords(q))

        return {
            "plan": plan_output.model_dump(),
            "optimized_queries": expanded_queries,
            "metadata_filter": chroma_filter,
            "retry_count": state.get("retry_count", 0) + 1
        }

    except Exception as e:
        logger.error(f"Failed to generate plan: {e}")
        raise e



