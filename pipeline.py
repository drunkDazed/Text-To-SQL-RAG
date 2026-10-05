import sqlite3
import re
import os
import threading
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from google import genai
from google.genai import types

# ==========================================
# 1. DATABASE SETUP & INITIALIZATION
# ==========================================
# check_same_thread=False + an explicit lock, because Streamlit can serve
# multiple sessions on shared threads and a single sqlite3 connection is
# not safe for concurrent cross-thread use.
conn = sqlite3.connect(':memory:', check_same_thread=False)
db_lock = threading.Lock()
cursor = conn.cursor()
cursor.execute("PRAGMA foreign_keys = ON;")

cursor.execute('''
CREATE TABLE PhotoObjAll (
    objID INTEGER PRIMARY KEY,
    ra REAL NOT NULL,
    dec REAL NOT NULL,
    u REAL, g REAL, r REAL, i REAL, z REAL,
    type INTEGER
);
''')

cursor.execute('''
CREATE TABLE SpecObjAll (
    specObjID INTEGER PRIMARY KEY,
    bestObjID INTEGER,
    class TEXT NOT NULL,
    subClass TEXT,
    z REAL,
    FOREIGN KEY(bestObjID) REFERENCES PhotoObjAll(objID)
);
''')

# Secondary indexes on the columns realistic astrophysics queries filter on.
# Without these, EVERY query that isn't a bare primary-key lookup triggers a
# full table scan, and the cost-optimization gatekeeper below would reject
# essentially all legitimate queries (redshift range, object class, magnitude
# cuts) even though they're correct.
cursor.execute("CREATE INDEX idx_photoobj_coords ON PhotoObjAll(ra, dec);")
cursor.execute("CREATE INDEX idx_photoobj_type ON PhotoObjAll(type);")
cursor.execute("CREATE INDEX idx_photoobj_g ON PhotoObjAll(g);")
cursor.execute("CREATE INDEX idx_specobj_class ON SpecObjAll(class);")
cursor.execute("CREATE INDEX idx_specobj_z ON SpecObjAll(z);")
cursor.execute("CREATE INDEX idx_specobj_bestobjid ON SpecObjAll(bestObjID);")

# A slightly richer mock dataset -- one row per table can't exercise
# filtering, joins, or the self-correction loop in any meaningful way.
_photo_rows = [
    (1, 180.05, -0.5, 18.2, 17.1, 16.5, 16.2, 16.0, 3),
    (2, 181.32, 1.2, 19.8, 19.1, 18.7, 18.4, 18.2, 3),
    (3, 179.88, -1.1, 21.4, 20.2, 19.5, 19.0, 18.7, 3),
    (4, 182.10, 0.3, 17.5, 16.9, 16.4, 16.1, 15.9, 6),
    (5, 178.44, 2.0, 22.1, 21.0, 20.1, 19.6, 19.2, 3),
]
_spec_rows = [
    (101, 1, 'QSO', 'BROADLINE', 2.15),
    (102, 2, 'QSO', 'BROADLINE', 2.85),
    (103, 3, 'QSO', 'NARROWLINE', 1.90),
    (104, 4, 'GALAXY', 'STARFORMING', 0.08),
    (105, 5, 'QSO', 'BROADLINE', 3.40),
]
cursor.executemany("INSERT INTO PhotoObjAll VALUES (?,?,?,?,?,?,?,?,?);", _photo_rows)
cursor.executemany("INSERT INTO SpecObjAll VALUES (?,?,?,?,?);", _spec_rows)
conn.commit()

# ==========================================
# 2. SCHEMA EXTRACTION & VECTOR ROUTER
# ==========================================
def extract_schema_metadata(db_cursor):
    db_cursor.execute("""
        SELECT name, sql FROM sqlite_schema 
        WHERE type='table' AND name NOT LIKE 'sqlite_%';
    """)
    tables = db_cursor.fetchall()
    extracted_docs = []
    for table_name, table_ddl in tables:
        formatted_doc = f"Table: {table_name}\nDDL: {table_ddl}"
        extracted_docs.append({'table': table_name, 'ddl': formatted_doc})
    return extracted_docs

schema_metadata = extract_schema_metadata(cursor)
schema_texts = [meta['ddl'] for meta in schema_metadata]

# Cosine similarity via normalized embeddings + Inner Product index
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
dimension = embedding_model.get_sentence_embedding_dimension()
vector_index = faiss.IndexFlatIP(dimension)

schema_embeddings = embedding_model.encode(schema_texts, normalize_embeddings=True)
vector_index.add(np.ascontiguousarray(schema_embeddings, dtype=np.float32))

def retrieve_schema_context(query_string: str, k: int = 2) -> str:
    # Cap k at the number of tables we actually have, otherwise faiss pads
    # results with -1 indices.
    k = min(k, len(schema_texts))
    query_emb = embedding_model.encode([query_string], normalize_embeddings=True)
    _, indices = vector_index.search(np.ascontiguousarray(query_emb, dtype=np.float32), k)

    retrieved = [schema_texts[idx] for idx in indices[0] if idx != -1]
    return "\n\n".join(retrieved)

# ==========================================
# 3. ROBUST PARSING & EXECUTION SANDBOX
# ==========================================
def extract_sql(llm_response: str) -> str | None:
    if not llm_response:
        return None
    # Flexible regex: handles optional sql tag, mixed casing, whitespace variations
    match = re.search(r"```(?:sql)?\s*\n?(.*?)\n?```", llm_response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: inspect if the raw string is an unfenced SELECT statement
    cleaned = llm_response.strip()
    if cleaned.upper().startswith(("SELECT", "WITH")):
        return cleaned
    return None

def execute_sql_sandboxed(query: str, max_rows: int = 100):
    if not query:
        return False, "Error: Extractor failed to parse a valid SQL string from LLM output."

    # AST / Hard-block sanity check
    normalized = query.strip().upper()
    if not normalized.startswith("SELECT") and not normalized.startswith("WITH"):
        return False, "Security Violation: Non-read-only statement detected."

    try:
        with db_lock:
            cursor.execute(query)
            results = cursor.fetchmany(max_rows)
        return True, results
    except sqlite3.Error as e:
        return False, f"SQLite Execution Error: {str(e)}"

# ==========================================
# 4. AGENTIC GENERATION & SELF-CORRECTION
# ==========================================

_api_key = os.environ.get("GEMINI_API_KEY")
if not _api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Export it in your environment "
        "(e.g. `export GEMINI_API_KEY=...`) rather than hardcoding it in source."
    )
client = genai.Client(api_key=_api_key)

# Sentinel so the retry loop can tell "LLM produced bad SQL" (a real
# self-correction case) apart from "the API call itself failed" (an infra
# problem that retrying the same prompt won't fix).
class LLMCallError(Exception):
    pass

def call_llm(prompt: str, role: str) -> str:
    if role in ["writer_initial", "writer_correction"]:
        system_instruction = (
            "You are an expert SQL developer. Output ONLY valid SQLite queries "
            "wrapped in ```sql ``` blocks. Do not include pleasantries or explanations."
        )
    elif role == "critic":
        system_instruction = (
            "You are a strict database debugging agent. Classify the execution error "
            "and output a precise, one-sentence correction strategy."
        )
    elif role == "synthesizer":
        system_instruction = (
            "You are a data analyst. Convert the raw database output into a concise, "
            "natural language answer that directly addresses the user's question. "
            "Do not explain the SQL or the database structure."
        )
    else:
        raise ValueError(f"Unknown LLM role: {role}")

    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
            ),
        )
        return response.text
    except Exception as e:
        # Re-raised as a distinct type so the pipeline can treat this as an
        # infra failure instead of spending a self-correction attempt on it.
        raise LLMCallError(str(e)) from e

def run_gatekeeper(sql_query: str) -> tuple[bool, str]:
    # Defensive check to trap empty or failed LLM outputs
    if not sql_query:
        return False, "Extraction Error: The LLM failed to output a valid SQL string wrapped in ```sql ``` blocks."

    normalized = sql_query.strip().upper()
    if not normalized.startswith("SELECT") and not normalized.startswith("WITH"):
        return False, "Security Violation: DML/DDL statements are strictly prohibited."

    try:
        with db_lock:
            cursor.execute(f"EXPLAIN QUERY PLAN {sql_query}")
            plan = cursor.fetchall()

        for row in plan:
            detail = row[3].upper()
            # Only block scans on tables large enough for it to matter.
            # On small/demo tables SQLite may legitimately prefer a scan
            # over an index seek, and treating every such scan as a hard
            # failure rejects correct queries outright. This should be
            # tightened again once the dataset reflects real SDSS table sizes.
            if "SCAN" in detail and "COVERING INDEX" not in detail and "USING INDEX" not in detail:
                table_match = re.search(r"SCAN (\w+)", detail)
                table_name = table_match.group(1) if table_match else None
                row_count = None
                if table_name:
                    with db_lock:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        row_count = cursor.fetchone()[0]
                if row_count is None or row_count > 10_000:
                    return False, (
                        f"Cost Optimization Block: Query initiates a full table scan "
                        f"({row[3]}). Rewrite the SQL to utilize indexed columns."
                    )

        return True, "Optimization Passed"
    except sqlite3.Error as e:
        return False, f"SQLite EXPLAIN Error: {str(e)}"

def run_agentic_pipeline(user_query: str, max_retries: int = 3):
    schema_context = retrieve_schema_context(user_query, k=2)
    conversation_history = []

    writer_prompt = (
        f"Database Schema:\n{schema_context}\n\n"
        f"User Query: {user_query}\n\n"
        "Generate an optimized SQL query. Format: wrap SQL in ```sql ... ```."
    )

    try:
        llm_output = call_llm(writer_prompt, "writer_initial")
    except LLMCallError as e:
        print(f"[Infra Failure] Initial writer call failed: {e}")
        return None
    sql_query = extract_sql(llm_output)

    for attempt in range(1, max_retries + 1):
        # 1. Gatekeeper Check
        gatekeeper_passed, gatekeeper_msg = run_gatekeeper(sql_query)

        if not gatekeeper_passed:
            print(f"[Attempt {attempt} Blocked by Gatekeeper] {gatekeeper_msg}")
            result = gatekeeper_msg
            success = False
        else:
            # 2. Sandbox Execution
            success, result = execute_sql_sandboxed(sql_query)

        if success:
            print(f"\n[DEBUG] Raw SQL Output: {result}")

            # 3. Output Synthesis
            synth_prompt = f"User Question: {user_query}\nDatabase Output: {result}\nSynthesize the final answer."
            try:
                final_answer = call_llm(synth_prompt, "synthesizer")
            except LLMCallError as e:
                print(f"[Infra Failure] Synthesizer call failed: {e}")
                final_answer = (
                    "The query executed successfully, but the answer could not be "
                    "synthesized due to an API error. Raw results are shown above."
                )
            print(f"\n--- FINAL AGENT RESPONSE ---\n{final_answer}\n")

            return {"query": sql_query, "raw_data": result, "final_answer": final_answer, "attempts": attempt}

        print(f"[Attempt {attempt} Failed] Triggering Critic Node...")

        critic_prompt = (
            f"Original Query: {user_query}\n"
            f"Schema:\n{schema_context}\n"
            f"Failed SQL: {sql_query}\n"
            f"Error Traceback: {result}\n"
            "Analyze the root cause and provide an explicit correction strategy."
        )
        try:
            critic_feedback = call_llm(critic_prompt, "critic")
        except LLMCallError as e:
            print(f"[Infra Failure] Critic call failed: {e}")
            break
        conversation_history.append({"attempt": attempt, "failed_sql": sql_query, "error": result, "critique": critic_feedback})

        correction_prompt = (
            f"Original Query: {user_query}\n"
            f"Schema:\n{schema_context}\n"
            f"Correction History: {conversation_history}\n"
            "Generate the corrected SQL query in ```sql ... ```."
        )
        try:
            llm_output = call_llm(correction_prompt, "writer_correction")
        except LLMCallError as e:
            print(f"[Infra Failure] Correction writer call failed: {e}")
            break
        sql_query = extract_sql(llm_output)

    print("[Pipeline Failure] Maximum retry threshold reached.")
    return None

if __name__ == "__main__":
    # This will only run if you execute pipeline.py directly in the terminal
    run_agentic_pipeline("Find right ascension and declination for quasars.")
