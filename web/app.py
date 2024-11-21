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
from dotenv import load_dotenv
import requests


load_dotenv()

app = Flask(__name__)
openai_key = os.getenv("OPEN_API_KEY")
TARGET_URL = "http://127.0.0.1:5000/query"



@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data['question']

    response = requests.post(TARGET_URL, json=data).json()

    print(response, '############################')
    # 응답과 함께 검색된 문서의 내용을 반환
    # print("검색된 문서의 내용: ")
    # for doc in response["source_document"]:
    #     print(doc.page_content[:200])
    return jsonify({"answer": response["answer"]})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True, port=8080)
