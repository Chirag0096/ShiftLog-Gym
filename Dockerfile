FROM python:3.11-slim

RUN apt-get update && apt-get install -y git build-essential

# Set up a non-root user for Hugging Face Spaces per guidelines
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user pyproject.toml .
COPY --chown=user shiftlog_gym ./shiftlog_gym
COPY --chown=user observatory ./observatory
COPY --chown=user train ./train
COPY --chown=user tests ./tests
COPY --chown=user app.py .
COPY --chown=user README.md .

RUN mkdir -p plots && chown user:user plots

# Install the package in editable mode to expose observatory
RUN pip install -e .[observatory]

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
