Agentic RAG Assistant: AWS Architectural Intelligence
📝 Project Description
This project is an Agentic Retrieval-Augmented Generation (RAG) assistant designed to answer complex architectural questions about AWS RAG options. It is strictly grounded in the AWS Prescriptive Guidance document: "Retrieval Augmented Generation options and architectures on AWS".

The system utilizes a multi-agent orchestration pattern (Planner, Retriever, and Synthesizer) built with LangGraph to ensure accurate, cited, and traceable answers.

⚙️ Prerequisites
Python 3.10 or higher

An active API Key (e.g., Groq API Key for LLM inference)

🚀 Setup & Installation
1. Clone and Prepare the Environment
Open your terminal in the project root folder:

Bash

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
2. Configure API Keys
The system requires an API key to function.

Create a file named .env in the root directory (trinity-assignment/).

Add your API key to the file:

Plaintext

GROQ_API_KEY=your_actual_api_key_here
🛠️ How to Run the Project
Follow these exact steps to start the assistant:

Step 1: Navigate to the Source Folder
From the root trinity-assignment directory, move into the src folder:

Bash

cd src
Step 2: Execute the Application
Run the main script using Python:

Bash

python main.py
Step 3: Enter Your Query
Once the application starts, you will see a prompt in your command line. Simply type your question about AWS RAG architectures and press Enter.

📂 Project Structure
Plaintext

trinity-assignment/
├── .env                  # Private API Keys (DO NOT SHARE)
├── requirements.txt      # Project dependencies
├── vectorstore/          # Persisted ChromaDB indices
├── data/
│   └── raw_documents/    # Source AWS PDF
└── src/                  # Application source code
    ├── main.py           # Entry point
    ├── agents/           # Planner, Retriever, and Synthesizer logic
    └── ingestion/        # PDF processing and vector indexing
✨ Key Features

Agentic Planning: A Planner agent analyzes your query to determine the best retrieval strategy.

Strict Grounding: The assistant will only answer based on the provided AWS document and will explicitly state if information is missing.

Structured Citations: Every response includes references to specific sections of the AWS guide.

Semantic Chunking: Documents are processed by headings and sections to maintain technical context
