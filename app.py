import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# =============================
# --- LLM and RAG Setup ---
# =============================

# Function that loads the document and creates the RAG pipeline
def create_rag_chain(document_path):
    # Load the document
    with open(document_path, 'r', encoding='utf-8') as f:
        document_text = f.read()

    # 1. Split the document into small "chunks"
    # This makes it easier for the model to find relevant information.
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=200,      # Chunk size (in characters)
        chunk_overlap=50,    # Overlap between chunks
        length_function=len
    )
    docs = text_splitter.split_text(document_text)

    # 2. Create "embedding vectors" for each chunk
    # Embeddings convert text into numerical vectors that computers can understand semantically.
    # all-MiniLM-L6-v2 is a small, fast model specialized in converting text into vectors.
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 3. Create a vector store (FAISS) to save and search embeddings
    # This is like creating a searchable index for our "textbook."
    db = FAISS.from_texts(docs, embeddings)

    # 4. Configure connection to the local LLM server (LM Studio)
    llm = ChatOpenAI(
        # ↓↓↓ Paste LM Studio's "API Identifier" here ↓↓↓
        model_name="local-model",            # Specify to use the local model
        base_url="http://p52:8001/v1", # Address of the LM Studio server
        api_key="not-needed",                # No API key needed for a local server
        temperature=0.1                      # Low temperature to stick to reference text for reliable answers
    )

    # 5. Create the RetrievalQA chain
    # This chain combines a retriever (FAISS index) with the LLM.
    # When given a query, it first finds the most relevant text chunks,
    # then passes them along with the query to the LLM to generate an answer.
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff" means stuffing all relevant chunks into the prompt
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain

# Create the RAG chain using knowledge.txt
rag_chain = create_rag_chain("the-boston-cooking-school-cookbook.txt")

# =============================
# --- Streamlit UI ---
# =============================

st.title("Ratatouille AI")
st.write("Lets get gooking")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Redisplay messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Respond to the user's input
if prompt := st.chat_input("Enter your question"):
    # Display the user's message
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add the user's message to history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get the LLM's response
    response = rag_chain.invoke({"query": prompt})
    answer = response["result"]

    # Display the assistant's response
    with st.chat_message("assistant"):
        st.markdown(answer)
    # Add the assistant's response to history
    st.session_state.messages.append({"role": "assistant", "content": answer})
