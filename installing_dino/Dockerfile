FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
     TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
     SETUPTOOLS_USE_DISTUTILS=stdlib

# Install libraries in the brand new image. 
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         build-essential \
         git \
         python3-opencv \
         ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory for all the subsequent Dockerfile instructions.
WORKDIR /opt/program

RUN git clone https://github.com/IDEA-Research/GroundingDINO.git

RUN mkdir GroundingDINO/weights && \
     cd GroundingDINO/weights && \
     wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth || true

ENV PATH=/usr/local/cuda/bin:$PATH

RUN python -m pip install --no-cache-dir \
          numpy==1.26.2 \
          torch==2.3.1+cu121 \
          torchvision==0.18.1+cu121 \
          --index-url https://download.pytorch.org/whl/cu121 && \
     python -m pip install --no-cache-dir \
          transformers==4.40.2 \
          addict==2.4.0 \
          yapf \
          timm==0.9.16 \
          opencv-python==4.9.0.80 \
          supervision>=0.22.0 \
          pycocotools==2.0.8 \
          pandas==2.2.1