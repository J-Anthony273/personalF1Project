FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/cache

ENV FASTF1_CACHE_DIR=/app/cache
ENV SECRET_KEY=f1Project

EXPOSE 5050

CMD ["gunicorn", "-b", "0.0.0.0:5050", "f1Project:app"]