FROM python:3.12.12-slim

WORKDIR /app

# Copy the project
COPY pyproject.toml uv.lock ./

# Install requirements
#COPY requirements.txt /app/requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uv
COPY . /app


RUN uv sync --frozen


# Run the server
# CMD ["gunicorn", "agents_server:app", "--bind", "0.0.0.0:8000", "--workers", "4"]
CMD ["uv", "run", "python", "agents_server.py"]
EXPOSE 8000