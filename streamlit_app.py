import torch
from transformers import pipeline

#patch to avoid streamlit crash with torch.classes
if not hasattr(torch.classes,"__path__"):
    torch.classes.__path__ = []

import streamlit as st #imports libraries as usual

generator = pipeline("text-generation", model="microsoft/DialoGPT-small") #generate model that holds short conversations using 350MB

#session state initialization and hold messages
if "messages" not in st.session_state:
    st.session_state.messages = []
st.title("Hi, there. I am your personal chatbot.")

#display previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])

#user input
prompt = st.chat_input("Enter a prompt.") #only allows input within context
if prompt:
    st.chat_message("user").markdown(prompt) #display current message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    #generate responses
    response = generator(
        prompt, #ensures response follows prompt
        max_length=50, #limits response length for avoided memory spikes
        do_sample=True, 
        truncation=True, #cut responses short if exceeds max_length
        temperature=0.7) #reduces randomness
    generated_text = response[0]["generated_text"]
    reply = generated_text[len(prompt):] #avoids repeating prompt
    
    #display replies
    st.chat_message("assistant").markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})

