FROM python:3.11-slim

# Instalacja zależności systemowych (graphviz + git)
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Kopiujemy tylko requirements żeby cache Dockera działał
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopiujemy resztę plików projektu
COPY . .

# Instalujemy kernel Jupyter
RUN python -m ipykernel install --user --name ml --display-name "Python (ml)"

EXPOSE 8888

CMD jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --IdentityProvider.token=''
