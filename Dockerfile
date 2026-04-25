FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

COPY . /app

RUN pip install --upgrade pip
RUN pip install -e .

EXPOSE 8000

CMD ["python", "-m", "server.app"]
