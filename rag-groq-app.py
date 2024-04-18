import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

## load the Groq API key
api_key = os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    embeddings = OllamaEmbeddings()
    loader = WebBaseLoader("https://docs.smith.langchain.com/")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    docs = loader.load()
    final_documents = text_splitter.split_documents(docs[:50])
    vectors = FAISS.from_documents(final_documents, embeddings)
    
    st.session_state.loader = loader
    st.session_state.embeddings = embeddings
    st.session_state.docs= docs

    st.session_state.text_splitter = text_splitter
    st.session_state.final_documents = final_documents
    st.session_state.vectors = vectors

st.title("ChatGroq Demo")

llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

prompt=st.text_input("Input you prompt here")

if prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({ "input": prompt })
    print("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")