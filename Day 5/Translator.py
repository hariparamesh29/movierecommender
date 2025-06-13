import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# --- Gemini API Key Setup ---
GOOGLE_API_KEY = "AIzaSyDzFx094teaEGzxkEEuNB5nmAqdnRW1D38"  # Replace with a valid Google API key

# --- Set up Gemini LLM via LangChain ---
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)  # Updated model name

# --- Define Prompt using ChatPromptTemplate ---
prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a translator that translates English to French. Respond with only the translated sentence."),
    ("human", "{input}")
])

# --- Create chain using LangChain ---
translation_chain = prompt | llm

# --- Streamlit UI ---
st.set_page_config(page_title="English to French Translator", layout="centered")
st.title("🌍 English to French Translator with Gemini")

# --- User Input ---
user_input = st.text_input("Enter a sentence in English:")

if st.button("Translate"):
    if not user_input.strip():
        st.warning("Please enter a valid English sentence.")
    else:
        try:
            # Run the LangChain chain
            response = translation_chain.invoke({"input": user_input})

            # Extract the content
            translated_text = response.content if hasattr(response, "content") else str(response)

            # Display result
            st.success("✅ Translation Successful!")
            st.markdown(f"**French:** {translated_text.strip()}")

        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")
