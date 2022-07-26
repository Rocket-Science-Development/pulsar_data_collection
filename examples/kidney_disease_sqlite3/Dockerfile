FROM python:3.9.13-slim-buster

WORKDIR /app

RUN adduser app && chown -R app /app
USER app

RUN python -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

COPY requirements.txt /app/requirements.txt
COPY main2.py /app/main.py
COPY kidney_disease.pkl /app/kidney_disease.pkl

RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["while true; do sleep 15; done"]
