FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade pip

COPY pyproject.toml README.md ./
COPY brain_tumor_fl ./brain_tumor_fl
COPY scripts ./scripts

RUN pip install --no-cache-dir .

COPY outputs ./outputs
COPY brain_tumor_mri ./brain_tumor_mri

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "brain_tumor_fl.web_app:app", "--host", "0.0.0.0", "--port", "8000"]
