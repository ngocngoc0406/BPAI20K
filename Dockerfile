# SafeDrive AI — CPU image (portable). For NVIDIA GPUs use an nvidia/cuda base and the cu118 torch wheels.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV SAFEDRIVE_DEVICE=cpu
EXPOSE 8501
# Mount trained models and (optionally) data:  -v %cd%/models:/app/models -v %cd%/data:/app/data -v %cd%/experiments:/app/experiments
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
