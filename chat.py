# PDF Question Answering ChatBot using LangChain, FAISS, and OpenAI

from itertools import chain  # Note: not used, consider removing
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI

# OpenAI API Key – fill this in securely 
OPENAI_API_KEY = ""

# Streamlit UI Header
st.header("Chat with Your PDF")

# Sidebar for file upload
with st.sidebar:
    st.title("Upload Documents")
    file = st.file_uploader("Upload a PDF file to begin", type="pdf")

# Only run the pipeline if a file is uploaded
if file is not None:
    # Step 1: Extract text from PDF
    reader = PdfReader(file)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text()

    # Step 2: Split the text into manageable chunks
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = splitter.split_text(full_text)

    # Step 3: Generate embeddings using OpenAI
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    # Step 4: Store the text chunks in a FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Step 5: Take user question as input
    user_question = st.text_input("Ask a question")

    if user_question:
        # Perform similarity search to retrieve relevant chunks
        matched_docs = vector_store.similarity_search(user_question)

        # Define LLM using OpenAI's ChatGPT
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

        # Load QA chain and generate response
        qa_chain = load_qa_chain(llm, chain_type="stuff")
        response = qa_chain.run(input_documents=matched_docs, question=user_question)

        # Display the answer
        st.write(response)
