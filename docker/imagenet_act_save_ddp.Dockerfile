# syntax=docker/dockerfile:1
# FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
# FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
# 기본 툴 + Python 3.11 설치 (PPA 없이)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-distutils \
        python3-pip \
        build-essential \
        git && \
    rm -rf /var/lib/apt/lists/*

# 3.11로 venv 생성
RUN python3.11 -m venv /opt/py311

# 이후부터는 /opt/py311 안의 python/pip을 기본으로 사용
ENV PATH="/opt/py311/bin:${PATH}"

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . ./

# RUN pip install --upgrade pip \
#     && pip install --no-cache-dir -r requirements.txt \
#     && pip install --no-cache-dir -e .

# 여기서부터는 무조건 python3 -m pip 사용
# RUN python -m pip install --upgrade pip && \
#     python -m pip install --no-cache-dir -r requirements.txt && \
#     python -m pip install --no-cache-dir -e .
RUN python -m pip install --no-cache-dir -e .
# ENTRYPOINT ["python", "tools/eval_imagenet_accuracy_ddp.py"]
ENTRYPOINT ["python", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "tools/save_backbone_activations.py"]
