FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

# Install Pyramid
RUN apt-get update && apt-get install -y git
RUN apt-get install -y cmake
RUN apt-get install -y --no-install-recommends libglib2.0-0
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir opencv-python-headless
RUN git clone https://github.com/sairajk/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection
ENV PYTHONPATH="/app/PyTorch-Pyramid-Feature-Attention-Network-for-Saliency-Detection:${PYTHONPATH}"

WORKDIR /app/src

# COPY pyramid/requirements.txt .
# RUN pip install --requirement requirements.txt
