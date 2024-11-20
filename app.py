from flask import Flask, request, jsonify, render_template
import os
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel  # pydantic v2 사용

app = Flask(__name__)
openai_key = os.getenv("OPENAI_API_KEY")

def load_pdf_and_split(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pdf_data = loader.load()
    print(f"PDF에서 로드된 문서 수: {len(pdf_data)}")  # 로드된 페이지 수 확인
    # for doc in pdf_data:
    #     print(doc.page_content[:200])  # 각 페이지의 첫 200자 확인
    text_splitter = TokenTextSplitter(chunk_size=2000, chunk_overlap=200)
    split_data = text_splitter.split_documents(pdf_data)
    print(f"분할된 청크 수: {len(split_data)}")  # 분할된 청크 수 확인
    return split_data

def setup_embeddings_and_chroma():
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    collection_name = "cau_collection"
    persist_directory = os.path.join(os.getcwd(), "cau_vect_embedding")
    pdf_path = os.path.join(os.getcwd(), "static", "cau.pdf")
    split_data = load_pdf_and_split(pdf_path)

    vectDB = Chroma(collection_name=collection_name, persist_directory=persist_directory, embedding_function=embeddings)

    # 데이터베이스 상태 확인
    if len(vectDB.get()) == 0:
        vectDB = Chroma.from_documents(split_data, embeddings, collection_name=collection_name, persist_directory=persist_directory)
        vectDB.persist()
        print("새로운 데이터를 벡터 데이터베이스에 저장했습니다.")
    else:
        print(f"벡터 데이터베이스에서 로드된 청크 수: {len(vectDB.get())}")

    return vectDB


def setup_conversational_chain():
    vectDB = setup_embeddings_and_chroma()
    retriever = vectDB.as_retriever(search_type='similarity', search_kwargs={'k': 5})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # $0.150 / 1M input tokens
    llm = ChatOpenAI(
        openai_api_key=openai_key,
        temperature=0.7,
        model_name="gpt-4o-mini"
    )
    chatQA = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return chatQA

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data['question']

    # 특정 질문에 대해 미리 정의된 응답
    if "무엇을 도와줄 수 있나요?" in question:
        return jsonify({"answer": "2024 중앙대학교 수시모집요강을 알려드릴 수 있습니다."})
    
    chatQA = setup_conversational_chain()
    response = chatQA({"question": question, "chat_history": []})

    # 응답과 함께 검색된 문서의 내용을 반환
    # print("검색된 문서의 내용: ")
    # for doc in response["source_document"]:
    #     print(doc.page_content[:200])

    return jsonify({"answer": response["answer"]})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
