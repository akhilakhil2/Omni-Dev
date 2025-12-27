"""
Synthesizer Agent Node for the RAG Workflow.

This module acts as the final reasoning engine. It takes the retrieved context 
and the original user query to generate a structured, professional response 
formatted in Markdown, following strict grounding rules to prevent hallucinations.
"""

import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from .state import AgentState

# --- LOGGING CONFIGURATION ---
logger = logging.getLogger("SynthesizerAgent")

def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Synthesizes the final answer based on retrieved documents and user intent.
    
    Args:
        state (AgentState): The current global state containing retrieved content.
        
    Returns:
        Dict[str, Any]: The generated response and a confidence score.
    """
    logger.info("Synthesizer Agent activated. Generating final response...")
    load_dotenv()

    # 1. Extract necessary data from state
    query = state.get("query")
    context = state.get("retriever_content")
    plan = state.get("plan", {})
    query_type = plan.get("query_type", "definition")
    sources = state.get("sources", [])

    # 2. Define Style Guidance based on Query Type
    style_guidance_map = {
        "comparison": "Use a clear, point-by-point comparison format. Highlight pros and cons of each.",
        "definition": "Provide a clear, concise definition and explain the core concept simply.",
        "recommendation": "Suggest the best AWS service for the user's needs and explain why it fits.",
        "trade-off": "Analyze the technical trade-offs, focusing on cost, complexity, and performance."
    }
    current_style = style_guidance_map.get(query_type, "Provide a helpful response based on the guide.")

    # 3. Configure the System Prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
         ### ROLE
        You are a Senior Technical Writer and AI Architect. Your goal is to synthesize 
        information from the AWS RAG Guide into a professional, grounded response.

        ### FORMATTING INSTRUCTIONS
        - **Comparison Format**: 
        If the query requires a comparison (e.g., "Managed vs Custom", "RAG vs Fine-Tuning"):
        1. Start with a brief high-level summary paragraph.
        2. Create a **Markdown Table** using the following structure:
            | Feature | [Service/Option A] | [Service/Option B] |
            | :--- | :--- | :--- |
            | **Key Characteristics** | description | description |
            | **Operational Overhead**| high/low/managed | high/low/managed |
            | **Flexibility** | level of control | level of control |
            | **Best For** | ideal use case | ideal use case |
        3. Follow the table with a deeper analysis of the trade-offs.

        - **Structure**: 
        - Use `##` for main headings.
        - Use `###` for sub-headings.
        - Separate all major sections with double line breaks.

        - **Scannability**: 
        - **Bold** all service names (e.g., **Amazon Kendra**, **Amazon Bedrock**).
        - Use `inline code` for technical parameters or API names.

        - **Grounding & Citations**:
        - Every technical claim must include a citation in the format: (see section: [Header Name]).
        - If information is missing from the provided context, state: "This information is not available in the provided AWS RAG guide."

        - **Final Section**: 
        - Always conclude with a `### Recommended Use Cases` section to provide actionable guidance based on the AWS document.

        ### RULES
        - **Strict Grounding**: Answer using ONLY the provided context. If the query does not match the context, or the context is insufficient, you MUST output exactly: "I don't know the answer. The provided document doesn't contain any information about this query."
        - **No Hallucination**: Do not use any external knowledge. If it is not in the context, it does not exist.
        - **Citations**: List the source pages at the very end under a '### Sources' heading: {source_list}.
        - **Architect Tone**: Maintain a formal, technical, and objective tone.

        QUERY TYPE: {query_type}
        STYLE GUIDANCE: {style_guidance}"""),
        
        ("human", "User Question: {user_query}\n\n### RETRIEVED CONTEXT\n{context}")
    ])

    # 4. Initialize the Model
    # Temperature is set to 0.2 for professional, consistent technical writing.
    model = ChatGroq(
        model='llama-3.3-70b-versatile', 
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2 
    )

    # 5. Execute Chain
    synthesizer_chain = prompt_template | model
    
    try:
        response = synthesizer_chain.invoke({
            "query_type": query_type,
            "style_guidance": current_style,
            "source_list": ", ".join(sources) if sources else "N/A",
            "user_query": query,
            "context": context
        })
        
        logger.info("Generation successful.")
        
        return {
            "generation": response.content,
            "confidence_score": 0.90  # Static score for this implementation
        }
        
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        return {
            "generation": "An error occurred during response generation.",
            "confidence_score": 0.0
        }

