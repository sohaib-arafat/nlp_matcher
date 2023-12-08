FROM python:3.11-slim


WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your Flask application and PyTorch model
COPY app.py .

# Optionally, copy any additional files your API requires

EXPOSE 8080

ENV FLASK_APP=app.py

CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
