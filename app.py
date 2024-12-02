import os
import streamlit as st
import tempfile
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Streamlit App Configuration
st.set_page_config(page_title="PDF Q&A Chatbot", page_icon="📚")

# Initialize session state for chat history and uploaded file
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'rag_chain' not in st.session_state:
    st.session_state.rag_chain = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = "default_session"

# Function to set up RAG chain
def setup_rag_chain(pdf_file):
    chroma_persist_dir = tempfile.mkdtemp(prefix="chroma_")
    # Temporary file handling
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    # Set up OpenAI API Key
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

    # LLM Configuration
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.5,
        max_tokens=150
    )

    # Embedding Model Configuration
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    # PDF Loader
    loader = PyPDFLoader(tmp_file_path)
    docs = loader.load()

    # Split Document into Chunks
    text_splitter = TokenTextSplitter(
        chunk_size=250,
        chunk_overlap=50
    )
    splits = text_splitter.split_documents(docs)

    # Semantic Search Retriever
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=chroma_persist_dir)
    vectorstore.persist()
    vectorstore_retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 2})

    # Keyword Search Retriever
    keyword_retriever = BM25Retriever.from_documents(splits, bm25params={'k1':2.0})

    # Create Ensemble Search
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retriever, keyword_retriever], 
        weights=[0.5, 0.5]
    )

    # Contextualized Prompt
    contextualized_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the history of the conversation and the user's most recent questions to provide accurate and relevant answers. Prioritize the user's latest query while incorporating historical context if it improves the response. Ensure clarity, relevance, and avoid redundancy."),
        MessagesPlaceholder('chat_history'),
        ("user", "{input}")
    ])

    history_aware_retriever = create_history_aware_retriever(llm, ensemble_retriever, contextualized_prompt)

    # System Prompt
    system_prompt = (
        "You are a knowledgeable assistant. Answer the user's questions based solely on the provided context. "
        "If you don't know the answer, clearly state that you don't know. Do not make up answers or speculate."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("user", "{input}")
    ])

    # Create RAG Chain
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    # Session History Management
    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # Conversational RAG Chain
    conversation_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    # Clean up temporary file
    os.unlink(tmp_file_path)

    return conversation_chain

# Streamlit App
def main():
    st.markdown(
    """
    <h1 style='text-align: center;'>📚 PDF Q&A Chatbot</h1>
    """,
    unsafe_allow_html=True
    )
    #st.title("📚 PDF Q&A Chatbot")

    # PDF Upload
    uploaded_file = st.file_uploader("Upload a PDF", type=['pdf'])

    # Sidebar for instructions
    st.sidebar.header("How to Use")
    st.sidebar.info(
        "1. Upload a PDF document\n"
        "2. Wait for the document to be processed\n"
        "3. Start asking questions about the document"
    )

    # Process PDF and set up RAG chain
    if uploaded_file is not None:
        try:
            # Setup RAG chain with uploaded PDF
            st.session_state.rag_chain = setup_rag_chain(uploaded_file)
            st.success("PDF processed successfully! You can now ask questions.")
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

    # Chat input
    if st.session_state.rag_chain is not None:
        user_query = st.chat_input("Ask a question about the document")

        # Display chat history
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.chat_message('user').write(message['content'])
            else:
                st.chat_message('assistant').write(message['content'])

        # Process user query
        if user_query:
            # Display user message
            st.chat_message('user').write(user_query)

            # Get RAG response
            try:
                response = st.session_state.rag_chain.invoke(
                    {"input": user_query},
                    config={"configurable": {"session_id": st.session_state.session_id}}
                )

                # Extract and display assistant response
                assistant_response = response['answer']
                st.chat_message('assistant').write(assistant_response)

                # Update chat history
                st.session_state.chat_history.append({
                    'role': 'user', 
                    'content': user_query
                })
                st.session_state.chat_history.append({
                    'role': 'assistant', 
                    'content': assistant_response
                })

            except Exception as e:
                st.error(f"Error generating response: {e}")
    else:
        st.info("Please upload a PDF to start chatting")

# Run the app
if __name__ == "__main__":
    main()
