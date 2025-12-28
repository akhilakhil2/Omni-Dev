
import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from .state import AgentState

# Initialize logger to track the final output generation phase
logger = logging.getLogger("SynthesizerAgent")

def synthesizer_node(state: AgentState) -> Dict[str, Any]:
    """
    Graph Node: Transforms retrieved context into a structured, grounded response.
    
    This node acts as a technical writer, using the 'intent' identified by the Planner 
    to apply specific formatting (like Markdown tables for comparisons). It enforces 
    strict grounding rules to prevent hallucinations and ensure all claims are cited.

    Args:
        state (AgentState): The current global state containing retrieved documents 
                            and the original plan.

    Returns:
        Dict[str, Any]: Updated state keys including the final generated text 
                        and a confidence score.
    """
    
    logger.info("Synthesizer Agent activated. Generating final response...")
    load_dotenv()

    # Step 1: Extract operational data from the state
    query = state.get("query")
    context = state.get("retriever_content")
    plan = state.get("plan", {})
    query_type = plan.get("query_type", "definition")
 
    # Step 2: Dynamic Style Mapping
    # We select specific instructions based on the query type to ensure the LLM 
    # uses the correct format (e.g., tables for comparisons).
    style_guidance_map = {
        "comparison": "Use a clear, point-by-point comparison format. Highlight pros and cons of each. Use tables to compare.",
        "definition": "Provide a clear, concise definition and explain the core concept simply.",
        "recommendation": "Suggest the best AWS service for the user's needs and explain why it fits.",
        "trade-off": "Analyze the technical trade-offs, focusing on cost, complexity, and performance."
    }

    current_style = style_guidance_map.get(query_type, "Provide a helpful response based on the guide.")

    # Step 3: Define the Synthesizer System Prompt
    # This prompt enforces the "Senior Architect" persona and strict grounding rules.
    prompt_template = ChatPromptTemplate.from_messages([
    ("system", """
### ROLE
You are a Senior Technical Writer and AI Architect specializing in AWS Cloud solutions. Your mission is to synthesize high-precision responses using ONLY the provided AWS RAG Guide context.

### RESPONSE STRATEGY
Based on the identified QUERY TYPE, apply the corresponding structure:
dont do duplicate contents.
1. **COMPARISON**: 
   - Start with a high-level summary.
   - Provide a Markdown Table comparing: **Key Characteristics**, **Operational Overhead**, **Flexibility**, and **Best For**.
   - Follow with a deep-dive analysis of technical trade-offs.

2. **RECOMMENDATION**: 
   - Define the relevant services.
   - Explicitly recommend the optimal AWS service for the query.
   - List detailed **Pros** and **Cons** for the recommended solution.

3. **DEFINITION**: 
   - Provide a detailed, clear explanation of the topic (2-3 sentences).
   - List practical use cases or technical applications.

4. **TRADE-OFF**: 
   - Define the concepts involved.
   - Analyze technical details focusing specifically on **Cost**, **Performance**, and **Complexity**.
   - Conclude with a summarized Pros/Cons breakdown.

### ARCHITECTURAL RULES (NON-NEGOTIABLE)
- **Strict Grounding**: Answer using ONLY the provided context. If the information is missing or insufficient, you MUST output exactly: "I don't know the answer. The provided document doesn't contain any information about this query."
- **Citations**: Every technical claim must be followed by a citation in this format: (see section: [Header Name]).
- **No Hallucinations**: Do not use external knowledge. If it's not in the context, it doesn't exist.
- **Scannability**: **Bold** all AWS service names. Use `inline code` for technical parameters or APIs.
- **Final Conclusion**: Every response must end with a `### Recommended Use Cases` section based on the AWS guide.

### CURRENT PARAMETERS
- **QUERY TYPE**: {query_type}
- **STYLE GUIDANCE**: {style_guidance}
"""),
    
    ("human", """User Question: {user_query}

### RETRIEVED CONTEXT
{context}

Generate the grounded response following the strategy for {query_type}:""")
])

    # Step 4: Initialize the Generation Model
    # A temperature of 0.2 is used to allow for professional phrasing without 
    # sacrificing factual accuracy.
    model = ChatGroq(
        model='llama-3.3-70b-versatile', 
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )
   

    synthesizer_chain = prompt_template | model
    
    try:
        # Step 5: Execute the generation chain
        response = synthesizer_chain.invoke({
            "query_type": query_type,
            "style_guidance": current_style,
            "user_query": query,
            "context": context
        })
        
        logger.info("Generation successful.")
        
        # Return the final content and a static validation score
        return {
            "generation": response.content,
            "confidence_score": 0.90 
        }
        
    except Exception as e:
        logger.error(f"Critical failure in Synthesizer Node: {e}")
        return {
            "generation": "An error occurred during response generation.",
            "confidence_score": 0.0
        }