# Implement RAG technique using Langchain, 
# ChromaDB on GPT 3.5 Turbo for conversational chatbot on a PDF document.

# 0. Setup virtual environment
# type the following commands in the terminal
# pip install virtualenv
# python3 -m venv venv
# source venv/bin/activate

# Create an OpenAI key
# https://platform.openai.com/apps.

# Install Dependencies
# pip install -r requirements.txt
# create an environment variable to store the OpenAI keys that are created in the last step.
# export OPENAI_API_KEY=<OPENAI-KEY>

# 1. Create Vector Embeddings from the User Manual PDF and store it in ChromaDB
## a. environment setup 
import os
import openai
import tiktoken
import chromadb

from langchain_community.document_loaders import OnlinePDFLoader, UnstructuredPDFLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import TokenTextSplitter
from langchain.memory import ConversationBufferMemory

## b. load the PDF document and split it into chunks
loader = PyPDFLoader("cau.pdf")
pdfData = loader.load()

text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
splitData = text_splitter.split_documents(pdfData)

## c. create vector embeddings from the user manual PDF and store it in ChromaDB
collection_name = "cau_collection"
local_directory = "cau_vect_embedding"
persist_directory = os.path.join(os.getcwd(), local_directory)

openai_key = 'use_own_ur_key'
embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
vectDB = Chroma.from_documents(splitData,
                      embeddings,
                      collection_name=collection_name,
                      persist_directory=persist_directory
                      )
vectDB.persist()

# 2. Create a Conversational Chatbot using RAG technique
## a. create a retriever from the vector embeddings
retriever = vectDB.as_retriever(
    search_type='similarity',
    search_kwargs={'k':5}
)

## b. setup the conversational chatbot
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
openai_key = 'OPENAI_API_KEY'
chatQA = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(openai_api_key=openai_key,
               temperature=0, model_name="gpt-3.5-turbo"), 
            retriever, 
            memory=memory)

## c. chat with the chatbot
chat_history = []
qry = ""
while qry != 'done':
    qry = input('Question: ')
    if qry != exit:
        response = chatQA.invoke({"question": qry, "chat_history": chat_history})
        print(response["answer"])