FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# Golden set, fixture world and recorded cassettes, so `eval --replay` works offline.
COPY evals ./evals
COPY prompts ./prompts

RUN useradd --create-home --uid 10001 leadgenie && chown -R leadgenie /app
USER leadgenie

# Pass ANTHROPIC_API_KEY at runtime (docker run -e / --env-file .env); never bake it in.
ENTRYPOINT ["leadgenie"]
CMD ["--help"]
