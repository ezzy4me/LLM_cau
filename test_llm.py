from langchain.chat_models import ChatOpenAI
import os

your_openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    openai_api_key=your_openai_api_key,
    model_name="gpt-3.5-turbo",
    temperature=0.0
)
print("LLM 객체 생성 성공:", llm)