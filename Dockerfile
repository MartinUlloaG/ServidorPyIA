FROM python:3.11-slim-bookworm

# Instalar las librerías del sistema requeridas para OpenCV y MediaPipe
# Agregamos libgles2 y libegl1 para solucionar la dependencia de OpenGL ES
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libgles2 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# Definir el directorio de trabajo
WORKDIR /app

# Copiar requirements.txt e instalar dependencias de Python de forma limpia
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del servidor (incluyendo archivos .task de MediaPipe)
COPY . .

# Comando de inicio usando la variable de puerto que Railway le asigne al contenedor
CMD ["sh", "-c", "uvicorn servidor:app --host 0.0.0.0 --port ${PORT:-8000}"]