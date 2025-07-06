FROM python:3.10-slim

# Crea una cartella per il codice
WORKDIR /app

# Copia il requirements e installa le dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il resto dei file dell'app
COPY . .

# Comando di default
CMD ["python", "analyze_yamnet.py"]