FROM continuumio/miniconda3

RUN pip install gunicorn streamlit pandas python-dotenv openai tiktoken scipy

COPY app /app

WORKDIR /app

CMD streamlit run --server.port 8501 front_app.py
