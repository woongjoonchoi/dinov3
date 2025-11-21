# syntax=docker/dockerfile:1
# FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY . ./

RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

# ENTRYPOINT ["python", "tools/eval_imagenet_accuracy_ddp.py"]
ENTRYPOINT ["python", "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "tools/eval_imagenet_accuracy_ddp.py"]
