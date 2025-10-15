import streamlit as st
from transformers import pipeline

# Load model
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="./neo_persuasion_finetuned")

generator = load_model()

st.title("Persuasive Text Generator")

prompt = st.text_input("Enter user statement:")

if prompt:
    response = generator(prompt, max_length=80, do_sample=True, temperature=0.8)
    st.write("**AI Response:**")
    st.write(response[0]['generated_text'])
