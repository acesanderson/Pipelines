# ML Pipelines Project

```
ml-pipelines/
├── docker-compose.yml
├── shared/
│   └── base_pipeline.py
├── pipelines/
│   ├── qwen-image/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── main.py
│   ├── whisper/
│   │   ├── Dockerfile
│   │   ├── pyproject.toml
│   │   └── main.py
│   └── video-gen/
│       ├── Dockerfile
│       ├── pyproject.toml
│       └── main.py
├── models/
└── outputs/
```

## Key Files

**shared/base_pipeline.py**
```python
from abc import ABC, abstractmethod

class BasePipeline(ABC):
    @abstractmethod
    def process(self, input_data): pass
    
    @abstractmethod
    def load_model(self): pass
```

**pipelines/qwen-image/main.py**
```python
from flask import Flask, request, jsonify
from shared.base_pipeline import BasePipeline
from diffusers import DiffusionPipeline
import torch, uuid, os

class QwenPipeline(BasePipeline):
    def load_model(self):
        self.model = DiffusionPipeline.from_pretrained("Qwen/Qwen-Image", torch_dtype=torch.bfloat16)
    
    def process(self, data):
        image = self.model(data['prompt']).images[0]
        filename = f"{uuid.uuid4().hex[:8]}.png"
        path = f"/app/outputs/{filename}"
        image.save(path)
        return {'image_path': path}

app = Flask(__name__)
pipeline = QwenPipeline()
pipeline.load_model()

@app.route('/generate', methods=['POST'])
def generate():
    return jsonify(pipeline.process(request.json))

if __name__ == '__main__': app.run(host='0.0.0.0', port=8000)
```

**docker-compose.yml**
```yaml
version: '3.8'
services:
  qwen-image:
    build: ./pipelines/qwen-image
    ports: ["8001:8000"]
    volumes: ["./models:/app/models", "./outputs:/app/outputs"]
  
  whisper:
    build: ./pipelines/whisper
    ports: ["8002:8000"]
    volumes: ["./models:/app/models", "./outputs:/app/outputs"]
```

**pipelines/qwen-image/Dockerfile**
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY shared/ /app/shared/
COPY pipelines/qwen-image/pyproject.toml .
RUN pip install -r pyproject.toml
COPY pipelines/qwen-image/ .
CMD ["python", "main.py"]
```

## Usage
```bash
docker-compose up -d
curl -X POST http://localhost:8001/generate -d '{"prompt":"cat"}' -H "Content-Type: application/json"
```
