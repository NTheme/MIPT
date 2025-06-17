FROM python:3.10-slim

RUN pip install --upgrade pip && pip install pipenv

WORKDIR /app

COPY . /app

RUN pipenv install
RUN mkdir -p /app/app/models

EXPOSE 8000
CMD ["pipenv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
