# AI 백엔드
FROM python:3.9-slim
WORKDIR /app

# 종속성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --ignore-installed

# 소스 코드 복사
COPY . .

# tell the port number the container should expose
EXPOSE 5000

# run the command
CMD ["python", "app.py"]
