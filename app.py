import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
import os
from dotenv import load_dotenv

load_dotenv()
###
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
search = DuckDuckGoSearchRun(name="Search")

st.title("LangChain - Chat with Search")
"""
In this example, using Streamlit to display the thoughts and actions.
"""
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if not api_key:
    api_key = os.getenv("GROQ_API_KEY")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web"}
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt := st.chat_input(placeholder="what is machine learning?"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="llama-3-8b-8192", streaming=True)
    response = llm.invoke(prompt)
    st.session_state["messages"].append({'role': 'assistant', 'content': response})
    st.chat_message("assistant").write(response)
