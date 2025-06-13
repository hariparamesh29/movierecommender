import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain.schema.output_parser import StrOutputParser
import tempfile

# --- HARD-CODED API KEY ---
GOOGLE_API_KEY = "AIzaSyDjAkmB5Bz5_oPiJA5k9zvbeQq2obOvv0o"  # Replace with your own key

# --- Streamlit UI ---
st.set_page_config(page_title="PDFbot", layout="centered")
st.title("📊 PDFbot")
st.write("Upload your PDF and ask questions about it.")

# --- Upload PDF ---
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    try:
        # --- 1. Load and Chunk PDF ---
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(pages)

        # --- 2. Create Embeddings and FAISS VectorStore ---
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        retriever = vectorstore.as_retriever()

        st.success("✅ Document processed successfully. You can now ask questions below.")

        # --- 3. LLM Setup (Gemini 2.0 Flash) ---
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=GOOGLE_API_KEY,
            temperature=0.2
        )

        # --- 4. Custom Prompt Template ---
        prompt = ChatPromptTemplate.from_template("""
You are an expert project management assistant. Use the following context to answer the user's question.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{question}
""")

        # --- 5. Runnable RAG Chain using Lambda to handle correct format ---
        chain = (
            {
                "context": RunnableLambda(lambda x: "\n\n".join(
                    doc.page_content for doc in retriever.get_relevant_documents(x["question"])
                )),
                "question": RunnableLambda(lambda x: x["question"])
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        # --- 6. Ask a Question ---
        question = st.text_input("Ask a question about your project document:")

        if question:
            with st.spinner("Thinking..."):
                try:
                    result = chain.invoke({"question": question})
                    st.subheader("📘 Answer:")
                    st.write(result)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    except Exception as e:
        st.error(f"❌ Failed to process the PDF: {str(e)}")
