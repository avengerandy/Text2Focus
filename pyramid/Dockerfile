FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

# Install python dependencies
RUN pip install --upgrade pip setuptools wheel
COPY ./pyramid/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Pyramid
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/sairajk/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection
ENV PYTHONPATH="/app/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection:${PYTHONPATH}"

WORKDIR /app/src

CMD ["python", "server.py"]
