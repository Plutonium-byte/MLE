FROM agrigorev/zoomcamp-model:2025

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH"

COPY "pyproject.toml" "uv.lock" "./"
RUN uv sync --locked

COPY "Q6.py" "./"

EXPOSE 9696

ENTRYPOINT ["uvicorn", "Q6:app", "--host", "0.0.0.0", "--port", "9696"]