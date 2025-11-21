FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# 필요한 패키지 설치 (torchvision)
RUN pip install --no-cache-dir torchvision scipy

WORKDIR /workspace

# 현재 디렉토리의 모든 파일을 컨테이너로 복사
COPY . .

# 기본 실행 커맨드
# (컨테이너 안에서 /workspace/imagenet_pytorch.py 실행)
CMD ["python", "imagenet_pytorch.py"]
