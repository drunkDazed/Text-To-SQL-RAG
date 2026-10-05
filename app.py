import streamlit as st
from pipeline import run_agentic_pipeline

st.set_page_config(page_title="Astrophysics SQL Agent", layout="centered")

st.title("Astrophysical Database Agent")
st.write("Natural language interface for relational sky survey data.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the database..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Executing agentic reasoning loop..."):
            response_data = run_agentic_pipeline(prompt)
        
        if response_data:
            with st.expander(f"Agent Diagnostic Trace (Resolved in {response_data['attempts']} attempts)"):
                st.code(response_data['query'], language="sql")
                st.write("**Raw Database Output:**")
                st.write(response_data['raw_data'])
                
            st.markdown(response_data['final_answer'])
            st.session_state.messages.append({"role": "assistant", "content": response_data['final_answer']})
        else:
            st.error("Pipeline failed. The Critic agent could not resolve the execution errors within the retry limit. Review the terminal logs.")