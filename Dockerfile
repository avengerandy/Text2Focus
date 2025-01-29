FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install opencv
RUN apt-get update && apt-get install -y cmake
RUN apt-get install -y --no-install-recommends libglib2.0-0
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir opencv-python-headless

# Install python dependencies
RUN pip install --no-cache-dir gradio
