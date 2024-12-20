FROM python:3.10.14

RUN apt-get update && apt-get install -y gcc\
    && rm -rf /var/lib/apt/lists/*


ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV PYTHONPATH=/app

WORKDIR /app

COPY requirements.txt /app/

RUN python3 -m pip install --upgrade pip && python3 -m pip install --no-cache-dir -r requirements.txt

COPY . /app/

ENTRYPOINT ["streamlit", "run" , "app.py", "--server.port=80"]

