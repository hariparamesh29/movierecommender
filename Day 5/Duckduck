import streamlit as st
from langchain.agents import initialize_agent, AgentType, Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
import os

# 💬 --- APP TITLE ---
st.set_page_config(page_title="🌐 Real-Time QA with Gemini + DuckDuckGo", layout="centered")
st.title("🤖 Ask Real-Time Questions with Gemini + 🌍 DuckDuckGo")

# 💡 --- API KEY (Hardcoded as requested) ---
GEMINI_API_KEY = "AIzaSyDjAkmB5Bz5_oPiJA5k9zvbeQq2obOvv0o"  # Replace with your actual key

# --- Initialize the model ---
try:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.7)
except Exception as e:
    st.error("❌ Error initializing Gemini model. Please check your API key.")
    st.stop()

# --- Set up search tool ---
search = DuckDuckGoSearchRun()
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Useful for answering questions about current events or recent facts.",
    )
]

# --- Initialize Agent ---
try:
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,  # No verbose terminal logs
        handle_parsing_errors=True
    )
except Exception as e:
    st.error("❌ Failed to initialize the agent.")
    st.stop()

# --- User Input ---
question = st.text_input("🔎 Enter your real-time question:", placeholder="e.g. What's the latest news on SpaceX?")
ask_button = st.button("🚀 Ask")

# --- Handle Query ---
if ask_button and question:
    with st.spinner("🤔 Thinking..."):
        try:
            response = agent.run(question)
            st.success("✅ Here's the answer:")
            st.write(response)
        except Exception as e:
            st.error(f"⚠️ Sorry, an error occurred while answering your question: {str(e)}")
