FROM python:3.11-slim

# libs de sistema que o mediapipe/opencv precisam
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN pip install --no-cache-dir mediapipe==0.10.14 fastapi "uvicorn[standard]" pillow numpy

# modelo oficial do Face Landmarker (478 pontos, ~4 MB)
ADD https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task /app/face_landmarker.task

COPY main.py .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
