import os
import streamlit as st
import pickle
import time
import langchain

from dotenv import load_dotenv
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL* {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # load data
    print("urls:", urls)
    loader = SeleniumURLLoader(urls=urls)
    main_placeholder.text("Data Loading....Started.... ✅✅✅")
    data = loader.load()

    print("data:", data)

    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter....Started.... ✅✅✅")
    docs = text_splitter.split_documents(data)

    print("docs:", docs)

    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building.... ✅✅✅")

    # save the FAISS index to a pickle file
    # with open(file_path, "wb") as f:
    #     pickle.dump(vectorstore_openai, f)

    vectorstore_openai.save_local(file_path)


query = st.text_input("Question: ")
query_clicked = st.sidebar.button("Get Answer")
if query_clicked:
    print("query:", query)
    if os.path.exists(file_path):
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(file_path, embeddings)
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)
        # {"answer": "", "source": []}

        st.header("Answer")
        st.subheader(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
