

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
    target_sections: List[str] = Field(
        description="The metadata keys (Header_2, Header_3, Header_4) mapped to each optimized query."
    )
    metadata_filter: List[str] = Field(
        description="The specific values within the target sections to filter the vector search."
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

    ### DOCUMENT HIERARCHY (Metadata Guide)
    The document metadata follows the format dict('Header_number':'value'). Here is the hierarchy:

    - Header_2: retrieval augmented generation options and architectures on aws
    - Header_2: generative ai options for querying custom documents
    - Header_2: fully managed retrieval augmented generation options on aws
    - Header_2: custom retrieval augmented generation architectures on aws
    - Header_2: choosing a retrieval augmented generation option on aws
    - Header_2: conclusion
    - Header_2: document history
    - Header_2: aws prescriptive guidance glossary
    - Header_3: intended audience
    - Header_3: objectives
    - Header_3: understanding retrieval augmented generation
    - Header_3: comparing retrieval augmented generation and fine-tuning
    - Header_3: use cases for retrieval augmented generation
    - Header_3: knowledge bases for amazon bedrock
    - Header_3: amazon q business
    - Header_3: amazon sagemaker ai canvas
    - Header_3: retrievers for rag workflows
    - Header_3: generators for rag workflows
    - Header_4: aws prescriptive guidance: retrieval augmented generation options and architectures on aws
    - Header_4: components of production-level rag systems
    - Header_4: data sources for knowledge bases
    - Header_4: vector databases for knowledge bases
    - Header_4: key features
    - Header_4: end-user customization
    - Header_4: amazon kendra
    - Header_4: amazon opensearch service
    - Header_4: amazon aurora postgresql and pgvector
    - Header_4: amazon neptune analytics
    - Header_4: amazon memorydb
    - Header_4: amazon documentdb
    - Header_4: pinecone
    - Header_4: mongodb atlas
    - Header_4: weaviate
    - Header_4: amazon bedrock
    - Header_4: sageMaker ai jumpstart
    ### PLANNER PLAN
    The PlannerPlan should be generated based on the following fields:
    
    1) **query_type**: Analyze the user's query and classify it into one of the following types: ["comparison", "definition", "recommendation", "trade-off"]. 
    - Return the selected `query_type`.

    2) **optimized_queries**: Break down the user query into more specific sub-queries. For example, if the query is "What is Amazon Kendra and what is Pinecone?", return the optimized queries as `['What is Amazon Kendra?', 'What is Pinecone?']`.

    3) **target_sections**: For each sub-query in `optimized_queries`, identify the relevant header section from the document hierarchy. Return the corresponding `Header_2`, `Header_3`, or `Header_4` sections for each sub-query. For example, for `['What is Amazon Kendra?', 'What is Pinecone?']`, return `['Header_4', 'Header_4']`. length of target_sections = length of optimized_queries

    4) **metadata_filter**: For each sub-query in `optimized_queries` and its corresponding target section in `target_sections`, return the exact metadata values associated with that section. For instance, for the sub-query `['What is Amazon Kendra?', 'What is Pinecone?']` and target sections `['Header_4', 'Header_4']`, return `['Amazon Kendra', 'Pinecone']` based on the document metadata. length of metadata_filters = length of target_sections

    5) **is_multi_pass**: If there is more than one sub-query in `optimized_queries`, set `is_multi_pass` to `True`. Otherwise, set it to `False`.

    6) **reasoning**: Explain the rationale behind the choices made for each field (`query_type`, `optimized_queries`, `target_sections`, `metadata_filter`, `is_multi_pass`). Provide an explanation for why each value was selected based on the user query.

    ### EXAMPLES
    - **Example 1:**
        User Query: "What is Amazon Kendra and what is Pinecone?"
        Feedback: None
        PlannerPlan:
        - `query_type`: "comparison"
        - `optimized_queries`: ["What is Amazon Kendra?", "What is Pinecone?"]
        - `target_sections`: ["Header_4", "Header_4"]
        - `metadata_filter`: ["Amazon Kendra", "Pinecone"]
        - `is_multi_pass`: True
        - `reasoning`: "The query asks for a comparison between Amazon Kendra and Pinecone, so it was classified as a 'comparison' query. Each query was broken down into specific topics, and both topics belong to 'Header_4', which refers to technical information about these tools. As there are two sub-queries, `is_multi_pass` is set to True."

    - **Example 2:**
        User Query: "What is Amazon Kendra?"
        Feedback: None
        PlannerPlan:
        - `query_type`: "definition"
        - `optimized_queries`: ["What is Amazon Kendra?"]
        - `target_sections`: ["Header_4"]
        - `metadata_filter`: ["Amazon Kendra"]
        - `is_multi_pass`: False
        - `reasoning`: "The query asks for a definition of Amazon Kendra, so it was classified as a 'definition' query. The query is straightforward and only requires one search term, which corresponds to 'Header_4' where Amazon Kendra is discussed. Since only one sub-query exists, `is_multi_pass` is set to False."

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