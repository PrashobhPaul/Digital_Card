
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="Prashobh's AI Assistant")

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    return tokenizer, model

tokenizer, model = load_model()

st.title("Ask Me Anything About Prashobh Paul")

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past_inputs" not in st.session_state:
    st.session_state.past_inputs = []

user_input = st.text_input("Your question", "")

if user_input:
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1) if st.session_state.chat_history_ids is not None else new_input_ids
    st.session_state.chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    st.session_state.past_inputs.append((user_input, response))

for q, a in reversed(st.session_state.past_inputs):
    st.markdown(f"**You:** {q}")
    st.markdown(f"**Prashobh's Assistant:** {a}")
