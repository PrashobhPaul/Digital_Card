
import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load GPT-2 model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

st.title("Prashobh's AI Assistant 🤖")
st.write("Ask me anything about Prashobh Paul and his work in AI!")

user_input = st.text_input("You:", "")

if user_input:
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, do_sample=True, top_k=50, top_p=0.95)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    st.markdown("**AI Assistant:**")
    st.write(response[len(user_input):].strip())
