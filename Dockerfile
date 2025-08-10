FROM python:3.9-slim

WORKDIR /app

# Copy requirements file and install dependencies
COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the mlruns directory and the main application files
COPY mlruns mlruns
COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

