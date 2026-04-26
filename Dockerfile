FROM python:3.11-slim

RUN apt-get update && apt-get install -y git build-essential curl

# Set up a non-root user for Hugging Face Spaces per guidelines
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

# Install astral-uv
COPY --from=ghcr.io/astral-sh/uv:0.4 /uv /uvx /bin/

# Create a virtual environment for uv to avoid system directory permission errors
RUN uv venv /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

COPY --chown=user requirements.txt .
RUN uv pip install --no-cache -r requirements.txt

COPY --chown=user pyproject.toml .
COPY --chown=user shiftlog_gym ./shiftlog_gym
COPY --chown=user observatory ./observatory
COPY --chown=user train ./train
COPY --chown=user tests ./tests
COPY --chown=user app.py .
COPY --chown=user README.md .

RUN mkdir -p plots && chown user:user plots

# Install the package seamlessly with uv
RUN uv pip install -e .[observatory]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
