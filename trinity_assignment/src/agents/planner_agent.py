

import os
import logging
from typing import List, Literal, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from .state import AgentState


# Initialize logger for tracking the Planner's decision-making process
logger = logging.getLogger("PlannerAgent")

class PlannerPlan(BaseModel):
    """
    Schema for the structured output of the Planner Agent.
    Defines how a user query is decomposed into a retrieval strategy.
    """
    
    query_type: Literal["comparison", "definition", "recommendation", "trade-off"] = Field(
        description="The classification of the user's intent to guide synthesis style."
    )
    optimized_queries: List[str] = Field(
        description="A list of atomic, searchable terms derived from the original user query."
    )
    is_multi_pass: bool = Field(
        description="Flag indicating if multiple distinct retrieval steps are required."
    )
    
    metadata_filter: List[dict] = Field(
        description="A list of metadata filters. Each filter is a dictionary where the key is the header level (e.g., 'Header_4') and the value is the specific metadata string."
    )
    reasoning: str = Field(
        description="The architectural justification for the selected retrieval strategy."
    )

def planner_node(state: AgentState) -> Dict[str, Any]:
    """
    Graph Node: Analyzes the user query and generates a structured retrieval plan.
    
    This node acts as the 'Director' of the RAG pipeline. It identifies the intent,
    maps queries to the AWS document hierarchy, and prepares metadata filters
    to ensure the Retriever only looks at relevant sections.

    Args:
        state (AgentState): The current global state of the LangGraph workflow.

    Returns:
        Dict[str, Any]: Updated state keys including the plan, queries, and incremented retry count.
    """
    
    logger.info("Planner Agent activated. Analyzing query...")

    # Load environment variables for API authentication
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
   
    
    if not groq_key:
        logger.error("GROQ_API_KEY not found in environment variables.")
        raise ValueError("Missing GROQ_API_KEY")

    # Initialize the LLM with structured output capabilities
    # We use a low temperature (0.1) to ensure consistent, deterministic planning logic.
    model = ChatGroq(model='llama-3.3-70b-versatile', api_key=groq_key, temperature=0.1)
    
    structured_llm = model.with_structured_output(PlannerPlan)
    
    

    # Construct the System Prompt with the Document Hierarchy (Grounding Logic)
    prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
    You are a Senior AI Architect specializing in Retrieval Augmented Generation (RAG) systems. 
    Your goal is to orchestrate a retrieval strategy for the retrieval agent node from the AWS RAG Guide.

    ### DOCUMENT HIERARCHY (Metadata Guide)
    The document metadata follows the format dict('Header_number':'value'). Here is the hierarchy:
    "Header_2":['generative ai options for querying custom documents', 'fully managed retrieval augmented generation options on aws', 'custom retrieval augmented generation architectures on aws', 'choosing a retrieval augmented generation option on aws', 'conclusion', 'document history'],
    "Header_3":['intended audience', 'objectives', 'understanding retrieval augmented generation', 'comparing retrieval augmented generation and fine-tuning', 'use cases for retrieval augmented generation', 'knowledge bases for amazon bedrock', 'amazon q business', 'amazon sagemaker ai canvas', 'retrievers for rag workflows', 'generators for rag workflows']
    "Header_4": ["components of production-level rag systems","data sources for knowledge bases","vector databases for knowledge bases","key_features","end-user customization","amazon kendra","amazon opensearch service","amazon aurora postgresql and pgvector","amazon neptune analytics","amazon memorydb","amazon documentdb","pinecone","mongodb atlas","weaviate","amazon bedrock","sageMaker ai jumpstart"]
      

    ### PLANNER PLAN
    The PlannerPlan should be generated based on the following fields:
    
    1) **query_type**: Analyze the user's query and classify it into one of the following types: ["comparison", "definition", "recommendation", "trade-off"]. 
    - Return the selected `query_type`.

    2) **optimized_queries**: Break down the user query into more specific sub-queries. For example, if the query is "What is Amazon Kendra and what is Pinecone?", return the optimized queries as `['What is Amazon Kendra?', 'What is Pinecone?']`.

    3) **metadata_filter**: For each sub-query in `optimized_queries`, you must generate a filter dictionary.
    - The format must be a List of dictionaries: `[{{"Header_2": "value1"}}, {{"Header_3": "value2"}}]`.
    - Use the DOCUMENT HIERARCHY (Metadata Guide) to find the exact string for "Header_2","Header_3","Header_4".
    - return only excact values matched to query in metadata  
    - All values must be in lowercase.
    - The number of dictionaries in the list MUST match the number of `optimized_queries`.

    ### EXAMPLE OUTPUT FORMAT
    If `optimized_queries` is ["What is Kendra?", "Tell me about Pinecone"]:
    `metadata_filter` = [{{"Header_4": "amazon kendra"}}, {{"Header_4": "pinecone"}}]  

    4) **is_multi_pass**: If there is more than one sub-query in `optimized_queries`, set `is_multi_pass` to `True`. Otherwise, set it to `False`.

    5) **reasoning**: Explain the rationale behind the choices made for each field (`query_type`, `optimized_queries`, `target_sections`, `metadata_filter`, `is_multi_pass`). Provide an explanation for why each value was selected based on the user query.


    ### NOTES:
    - DO NOT HALLUCINATE information. Refer only to the provided document metadata and user query.
    - The system must perform its tasks strictly based on the user's query and the document metadata without making any assumptions.
    """),

    ("human", """User Query: {query}
    Feedback: {revision_notes}
    
    Generate the PlannerPlan.""")
])

    # Build the execution chain
    planner_chain = prompt_template | structured_llm
    
    try:
        # Invoke the LLM to generate the plan
        plan_output = planner_chain.invoke({
            "query": state.get("query"),
            "revision_notes": state.get("revision_notes") or "Initial attempt."
        })
        
        logger.info(f"Plan generated successfully. Intent identified as: {plan_output.query_type}")
        
        # Return updates to the AgentState
        return {
            "plan": plan_output.model_dump(),             # Serialize Pydantic object to dict for state persistence
            "optimized_queries": plan_output.optimized_queries,
            "retry_count": state.get("retry_count", 0) + 1 # Increment retry counter to prevent infinite loops
        }
        
    except Exception as e:
        logger.error(f"Critical failure in Planner Node: {e}")
        raise e