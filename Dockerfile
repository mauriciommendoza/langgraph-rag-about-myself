FROM python:3.13-slim

# Copiar el binario de uv directamente
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias primero
COPY pyproject.toml uv.lock ./

# Instalar dependencias usando uv
# No usamos --no-dev por si las dependencias están mezcladas
RUN uv sync --frozen

# Copiar el resto del código
COPY . .

# Hugging Face expone el puerto 7860 por defecto
EXPOSE 7860

# Comando para ejecutar la aplicación
CMD ["uv", "run", "chainlit", "run", "app_ui.py", "--host", "0.0.0.0", "--port", "7860"]
