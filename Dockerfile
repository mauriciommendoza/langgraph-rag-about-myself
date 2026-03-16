FROM python:3.13-slim

# Copy uv binary directly
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory
WORKDIR /app

# Copy dependency files first (for Docker layer caching)
COPY pyproject.toml uv.lock .python-version ./

# Install dependencies
RUN uv sync --frozen

# Copy the rest of the source code
COPY . .

# Hugging Face Spaces exposes port 7860 by default
EXPOSE 7860

# Run the Chainlit application
CMD ["uv", "run", "chainlit", "run", "app_ui.py", "--host", "0.0.0.0", "--port", "7860"]
