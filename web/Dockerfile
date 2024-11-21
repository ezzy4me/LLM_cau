# 프론트엔드
FROM python:3.9-slim
WORKDIR /app

# 종속성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --ignore-installed

# 소스 코드 복사
COPY . .

# Flask 웹 서버 실행
CMD ["python", "app.py"]
