import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="Prashobh's AI Assistant", layout="centered")

st.title("Ask Me Anything")
st.markdown("This assistant can answer questions about my professional background and AI work.")

# Load a lightweight model
qa = pipeline("text-generation", model="tiiuae/falcon-rw-1b")

# Define static context about Prashobh
context = """
Prashobh Paul is an AI professional with 10+ years of experience, specialized in conversational systems and agentic frameworks.
He has worked with companies including TechMahindra, Cisco, and CIA on NLP, GenAI, and multi-agent AI solutions.
Some of his notable projects include calendar planning assistants and customer intelligence alert systems.
"""

# Input box
query = st.text_input("What would you like to know?")

if query:
    prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
    response = qa(prompt, max_new_tokens=60, do_sample=True, temperature=0.7)
    st.write(response[0]['generated_text'].split("Answer:")[-1])
