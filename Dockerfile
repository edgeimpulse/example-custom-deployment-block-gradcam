FROM public.ecr.aws/docker/library/python:3.10

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt ./
RUN pip3 --no-cache-dir install -r requirements.txt

COPY gradcam.py ./

ENTRYPOINT [ "python3", "-u", "gradcam.py" ]