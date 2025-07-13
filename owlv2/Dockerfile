FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# Install python dependencies
RUN pip install --upgrade pip setuptools wheel
COPY ./owlv2/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "server.py"]
