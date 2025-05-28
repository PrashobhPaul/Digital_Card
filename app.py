import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="Prashobh's AI Assistant", layout="centered")

st.title("🤖 Welcome to Prashobh's AI Assistant")
st.markdown("Ask me anything about my work, expertise, or experience!")

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# User input
user_input = st.text_input("Your question:", "")

# Generate response
if user_input:
    inputs = tokenizer.encode(user_input, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.markdown(f"**AI Assistant:** {response}")
