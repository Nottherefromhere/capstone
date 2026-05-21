FROM python:3.10-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_all_tests.sh .
COPY src/ ./src/
COPY tests/ ./tests/

RUN chmod +x run_all_tests.sh
RUN ./run_all_tests.sh

EXPOSE 5000
ENV FLASK_APP=src.app:app
ENV FLASK_RUN_HOST=0.0.0.0
CMD ["flask", "run", "--port", "5000"]
