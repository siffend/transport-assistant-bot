FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
# по умолчанию поднимается веб-демо; для бота: docker run ... python run_bot.py
EXPOSE 8000
CMD ["python", "run_web.py"]
