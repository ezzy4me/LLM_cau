# OpenAI API 키 설정
from flask import Flask, request, jsonify, render_template 
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
import chromadb

# Flask 앱 생성
app = Flask(__name__)
openai_key= 'write own ur openai key'

# PDF 로딩 및 텍스트 분할 함수
def load_pdf_and_split(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pdf_data = loader.load()
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=100)
    return text_splitter.split_documents(pdf_data)

# ChromaDB에 벡터 임베딩 저장
def setup_embeddings_and_chroma():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    
    collection_name = "cau_collection"
    local_directory = "cau_vect_embedding"
    persist_directory = os.path.join(os.getcwd(), local_directory)

    # PDF 로딩 및 임베딩 저장
    pdf_path = os.path.join(os.getcwd(), "static", "cau.pdf")  # PDF 파일 경로 설정
    split_data = load_pdf_and_split(pdf_path)  # PDF 로드 및 분할
    
    vectDB = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=embeddings)
    
    if len(vectDB.get()) == 0:  # 데이터가 없을 경우 벡터 스토어에 새로 임베딩 저장
        vectDB = Chroma.from_documents(split_data, embeddings, collection_name=collection_name, persist_directory=persist_directory)
        vectDB.persist()
    
    return vectDB

# 챗봇 체인 설정 함수
def setup_conversational_chain():
    vectDB = setup_embeddings_and_chroma()
    
    # 벡터 검색 기능 (retriever) 생성
    retriever = vectDB.as_retriever(search_type='similarity', search_kwargs={'k': 5})
    
    # 대화형 메모리 설정
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    # RAG 기반 챗봇 설정
    chatQA = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(openai_api_key=openai_key, temperature=0, model_name="gpt-3.5-turbo"), 
                retriever, 
                memory=memory)
    
    return chatQA

# Flask 라우트
@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data['question']
    
    # 특정 질문에 대해 미리 정의된 응답
    if "무엇을 도와줄 수 있나요?" in question:
        return jsonify({"answer": "2024 중앙대학교 수시모집요강을 알려드릴 수 있습니다."})
    
    # 챗봇 체인 호출
    chatQA = setup_conversational_chain()
    response = chatQA.invoke({"question": question, "chat_history": []})
    
    return jsonify({"answer": response["answer"]})

# 홈 페이지를 위한 기본 경로 (HTML 파일 로드)
@app.route('/')
def home():
    return render_template('index.html')  # HTML 파일 렌더링

if __name__ == "__main__":
    app.run(debug=True)
