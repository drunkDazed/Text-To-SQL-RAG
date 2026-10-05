# Autonomous Text-to-SQL RAG Agent

An autonomous, self-correcting Text-to-SQL pipeline designed to securely query relational databases using Retrieval-Augmented Generation (RAG). 

Standard zero-shot Text-to-SQL implementations suffer from context bloat, hallucinated schema aliases, and catastrophic database locking caused by unvalidated queries. This project re-architects the workflow into a closed-loop agentic system. It dynamically retrieves localized schema definitions via a local vector index, generates deterministic SQL, enforces static security analysis before execution, and autonomously debugs its own runtime errors.

## System Architecture

The pipeline executes through five distinct stages per user query:

1. Semantic Schema Retrieval (FAISS): Embeds the natural language query using a local sentence-transformers model and performs an exact inner-product vector search (IndexFlatIP) against the database schema. Injects only the top-K relevant table and column definitions into the prompt, mitigating context bloat.
2. Deterministic Generation (Writer): A Writer agent (powered by Gemini API) generates SQLite-compliant syntax using greedy decoding (Temperature = 0.0) to ensure mathematically rigid syntax compliance.
3. Programmatic Gatekeeper: An automated static analysis node intercepts the generated SQL before it touches the live database. It blocks destructive DDL/DML commands and evaluates the SQLite query optimizer (EXPLAIN QUERY PLAN) to reject brute-force, unindexed full table scans (SCAN).
4. Autonomous Self-Correction (Critic): If the Gatekeeper rejects the query or the SQLite engine throws a runtime error, a Critic agent intercepts the execution traceback. The traceback is sanitized and truncated to prevent context window explosion, then fed back to the Writer agent for autonomous regeneration (up to 3 retries).
5. Synthesis & Observability (Streamlit): Validated raw database tuples are processed by a Synthesizer agent into human-readable prose. The Streamlit frontend provides a dual-view UI: clean natural language for the user, and an expandable administrative trace exposing the exact SQL iterations and runtime exceptions for developer observability.

## Tech Stack

- LLM Orchestration: Gemini API (gemini-3.6-flash)
- Vector Search & Embeddings: FAISS (faiss-cpu), sentence-transformers/all-MiniLM-L6-v2
- Database Engine: SQLite (sqlite3)
- Frontend Interface: Streamlit
- Data Processing & ETL: Pandas

## Dataset: Sloan Digital Sky Survey (SDSS)

To validate the pipeline against complex relational algebra, the underlying database contains a normalized subset of SDSS astrophysical data.
- PhotoObj: Contains spatial coordinates (ra, dec) and raw photometric light magnitudes (u, g, r, i, z).
- SpecObj: Contains derived cosmological classifications (STAR, GALAXY, QSO) and spectroscopic redshift velocities.
- Domain Challenges Handled: The RAG index is specifically configured to handle domain anomalies, such as astronomical missing values hardcoded as -9999 rather than standard SQL NULL values, requiring the agent to utilize context-aware filtering.

## Installation & Setup

1. Clone the repository:
   git clone https://github.com/yourusername/text-to-sql-agent.git
   cd text-to-sql-agent

2. Create a virtual environment and install dependencies:
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt

3. Configure API Keys:
   Create a .env file in the root directory and add your Gemini API key:
   GEMINI_API_KEY="your_api_key_here"

4. Run the Application:
   streamlit run app.py

## Key Engineering Decisions

- FAISS IndexFlatIP over Cloud Vector Stores: Eliminated HTTP network latency and external cloud dependencies. Because a relational schema consists of hundreds of text vectors rather than millions, utilizing exact brute-force dot product calculations guarantees 100% retrieval recall without the accuracy loss associated with Approximate Nearest Neighbor (ANN) clustering.
- Traceback Truncation Protocol: Prevented context window explosion during the self-correction loop. Standard Python sqlite3 exceptions generate verbose multi-line stack traces. The pipeline truncates these to the terminal engine error string, reducing retry token consumption by over 80% while preserving the exact diagnostic signal for the Critic agent.
- Separation of Agent Personas: Partitioned the workflow into distinct Writer, Critic, and Synthesizer prompts. This prevents instruction dilution, ensuring the LLM acts strictly as a deterministic code generator during generation, a DBA during debugging, and a business analyst during user presentation.
